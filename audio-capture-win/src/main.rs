//! `vinanote-audiocap.exe`: Windows (WASAPI) drop-in for the macOS helper
//! `audio-capture/audiocap.swift`. The Tauri backend spawns it with piped stdin/stdout.
//!
//! CLI
//!   --check                                   print one `permissions` line, exit 0
//!   [--system] [--mic] [--exclude-pid <PID>]  capture until `stop` or stdin EOF
//!
//! stdout: one JSON object per line (UTF-8, `\n`, flushed per line):
//!   {"type":"audio","source":"system"|"microphone","ts":<unix s>,"rms":<f32>,"pcm_b64":"<base64 f32le>"}
//!       always exactly 1600 samples = 100 ms of 16 kHz mono f32; forwarded verbatim to the AI worker
//!   {"type":"status","source":..,"state":"capturing"[,"note":"loopback-no-exclusion"]}
//!   {"type":"error","source":..,"message":"<Vietnamese, shown verbatim by the UI>"}
//!   {"type":"ready","system":<bool>,"microphone":<bool>}   once, after the start-up attempts
//!   {"type":"stopped"}                                      right before exiting after a stop
//!   {"type":"permissions","microphone":"granted"|"denied","system_audio":"granted"|"denied"}
//!
//! stdin: one JSON object per line, {"cmd":"mute_mic"|"unmute_mic"|"mute_system"|"unmute_system"|"stop"}.
//!   Muted = that source's samples are dropped. `stop` or EOF => stop, emit `stopped`, exit 0.
//!   Anything else is ignored.

use serde_json::Value;
use std::io::{self, Write};

/// Write one JSON object as one `\n`-terminated line and flush. Holding stdout's lock for the
/// whole line is the mutex that keeps lines from the capture threads from interleaving.
fn emit(v: &Value) {
    let mut line = v.to_string();
    line.push('\n');
    let result = {
        let mut out = io::stdout().lock();
        out.write_all(line.as_bytes()).and_then(|()| out.flush())
    };
    if result.is_err() {
        std::process::exit(0);
    }
}

#[cfg(not(windows))]
fn main() {
    emit(
        &serde_json::json!({"type": "error", "source": "system", "message": "vinanote-audiocap.exe chỉ chạy trên Windows"}),
    );
    std::process::exit(1);
}

#[cfg(windows)]
fn main() {
    std::process::exit(win::run());
}

/// Platform-independent PCM plumbing, unit-tested on any host with `cargo test`.
#[cfg_attr(not(windows), allow(dead_code))]
mod pcm {
    use base64::prelude::*;
    use serde_json::{json, Value};
    use std::f64::consts::{PI, TAU};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub const SAMPLE_RATE: u32 = 16_000;
    /// Samples per `audio` line: 100 ms at 16 kHz.
    pub const CHUNK_SAMPLES: usize = 1_600;

    /// Re-chunks a 16 kHz mono stream into exact 1600-sample chunks, like `PCMChunkAccumulator`
    /// in the Swift helper: leftover samples carry over to the next push.
    #[derive(Default)]
    pub struct ChunkAccumulator {
        buf: Vec<f32>,
    }

    impl ChunkAccumulator {
        pub fn push(&mut self, samples: &[f32], mut emit: impl FnMut(&[f32])) {
            self.buf.extend_from_slice(samples);
            let whole = self.buf.len() - self.buf.len() % CHUNK_SAMPLES;
            self.buf[..whole]
                .chunks_exact(CHUNK_SAMPLES)
                .for_each(&mut emit);
            self.buf.drain(..whole);
        }
    }

    pub fn rms(chunk: &[f32]) -> f32 {
        if chunk.is_empty() {
            return 0.0;
        }
        (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt()
    }

    /// Standard base64 (with padding) of the samples as little-endian f32, which is what the AI
    /// worker decodes with `np.frombuffer(..., dtype="<f4")`.
    pub fn encode_pcm(chunk: &[f32]) -> String {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        BASE64_STANDARD.encode(bytes)
    }

    /// The `audio` line for one chunk, stamped with the emission time.
    pub fn audio_line(source: &str, chunk: &[f32]) -> Value {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        json!({"type": "audio", "source": source, "ts": ts, "rms": rms(chunk), "pcm_b64": encode_pcm(chunk)})
    }

    /// WASAPI loopback delivers no packets while nothing plays, but the VAD downstream needs a
    /// continuous 16 kHz clock (macOS delivers zeros). Once 100 ms behind, account for the whole
    /// deficit but return at most one second of zeros to append.
    pub fn silence_padding(elapsed: Duration, delivered: &mut u64) -> usize {
        let due = (elapsed.as_micros() * SAMPLE_RATE as u128 / 1_000_000) as u64;
        let deficit = due.saturating_sub(*delivered);
        if deficit < CHUNK_SAMPLES as u64 {
            return 0;
        }
        *delivered = due;
        deficit.min(u64::from(SAMPLE_RATE)) as usize
    }

    #[derive(Clone, Copy, Debug)]
    pub enum SampleFormat {
        F32,
        I16,
        /// 32-bit container, including 24 valid bits (left-justified, so the same scale applies).
        I32,
    }

    /// Device frames (interleaved, any rate and channel count) to 16 kHz mono f32 in [-1, 1].
    pub struct Converter {
        format: SampleFormat,
        channels: usize,
        resampler: Option<Resampler>,
    }

    impl Converter {
        pub fn new(format: SampleFormat, channels: usize, rate: u32) -> Self {
            let resampler = (rate != SAMPLE_RATE).then(|| Resampler::new(rate));
            Self {
                format,
                channels: channels.max(1),
                resampler,
            }
        }

        /// Appends the converted samples of `bytes` (whole frames, as WASAPI delivers them).
        pub fn push(&mut self, bytes: &[u8], out: &mut Vec<f32>) {
            let (format, channels) = (self.format, self.channels);
            let width = match format {
                SampleFormat::I16 => 2,
                SampleFormat::F32 | SampleFormat::I32 => 4,
            };
            let mono = bytes.chunks_exact(width * channels).map(|frame| {
                let sum: f32 = frame.chunks_exact(width).map(|b| decode(format, b)).sum();
                (sum / channels as f32).clamp(-1.0, 1.0)
            });
            match &mut self.resampler {
                None => out.extend(mono),
                Some(r) => r.process(&mut mono.collect::<Vec<_>>(), out),
            }
        }
    }

    fn decode(format: SampleFormat, b: &[u8]) -> f32 {
        match format {
            SampleFormat::F32 => {
                let sample = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                if sample.is_finite() {
                    sample
                } else {
                    0.0
                }
            }
            SampleFormat::I16 => i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0,
            SampleFormat::I32 => {
                i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
            }
        }
    }

    const FIR_TAPS: usize = 63;
    const CUTOFF_HZ: u32 = 7_000;

    /// Hamming-windowed sinc with a circular history so filtering continues across packets.
    struct LowPass {
        taps: [f32; FIR_TAPS],
        history: [f32; FIR_TAPS],
        next: usize,
    }

    impl LowPass {
        fn new(rate: u32) -> Self {
            let cutoff = f64::from(CUTOFF_HZ) / f64::from(rate);
            let center = (FIR_TAPS / 2) as f64;
            let taps: [f64; FIR_TAPS] = std::array::from_fn(|i| {
                let offset = i as f64 - center;
                let sinc = if offset == 0.0 {
                    2.0 * cutoff
                } else {
                    (TAU * cutoff * offset).sin() / (PI * offset)
                };
                let window = 0.54 - 0.46 * (TAU * i as f64 / (FIR_TAPS - 1) as f64).cos();
                sinc * window
            });
            let gain: f64 = taps.iter().sum();
            Self {
                taps: taps.map(|tap| (tap / gain) as f32),
                history: [0.0; FIR_TAPS],
                next: 0,
            }
        }

        fn filter(&mut self, sample: f32) -> f32 {
            self.history[self.next] = sample;
            self.next = (self.next + 1) % FIR_TAPS;
            self.history[self.next..]
                .iter()
                .chain(&self.history[..self.next])
                .zip(self.taps.iter())
                .map(|(sample, tap)| sample * tap)
                .sum()
        }
    }

    /// Streaming low-pass filter and linear interpolation to 16 kHz. The exact integer phase
    /// prevents drift. `pos` is the next output position in input samples times SAMPLE_RATE,
    /// counted from `prev` (the previous call's last sample, index 0 of [prev, input..]).
    pub struct Resampler {
        in_rate: u64,
        pos: u64,
        prev: f32,
        low_pass: Option<LowPass>,
    }

    impl Resampler {
        pub fn new(in_rate: u32) -> Self {
            Self {
                in_rate: in_rate.max(1) as u64,
                pos: 0,
                prev: 0.0,
                // Inputs with a Nyquist frequency at or below the cutoff are already band-limited.
                low_pass: (in_rate > 2 * CUTOFF_HZ).then(|| LowPass::new(in_rate)),
            }
        }

        pub fn process(&mut self, input: &mut [f32], out: &mut Vec<f32>) {
            if let Some(filter) = &mut self.low_pass {
                for sample in input.iter_mut() {
                    *sample = filter.filter(*sample);
                }
            }
            let Some(&last) = input.last() else { return };
            let scale = SAMPLE_RATE as u64;
            let end = input.len() as u64 * scale;
            let prev = self.prev;
            let at = |i: u64| if i == 0 { prev } else { input[i as usize - 1] };
            while self.pos < end {
                let i = self.pos / scale;
                let frac = (self.pos % scale) as f32 / scale as f32;
                let (a, b) = (at(i), at(i + 1));
                // The filter can overshoot on sharp transitions.
                out.push((a + (b - a) * frac).clamp(-1.0, 1.0));
                self.pos += self.in_rate;
            }
            self.pos -= end;
            self.prev = last;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn accumulator_emits_exact_chunks_and_carries_leftover() {
            let input: Vec<f32> = (0..4_000).map(|i| i as f32).collect();
            let mut acc = ChunkAccumulator::default();
            let mut chunks: Vec<Vec<f32>> = Vec::new();
            for part in [&input[..1_000], &input[1_000..2_000], &input[2_000..]] {
                acc.push(part, |c| chunks.push(c.to_vec()));
            }
            assert_eq!(chunks.len(), 2); // 4000 = 2 * 1600 + 800 carried
            assert!(chunks.iter().all(|c| c.len() == CHUNK_SAMPLES));
            assert_eq!(chunks[1][0], 1_600.0); // seamless across pushes
            acc.push(&input[..800], |c| chunks.push(c.to_vec()));
            assert_eq!(chunks.len(), 3);
            assert_eq!(chunks[2][0], 3_200.0); // the carried leftover comes first
            assert_eq!(chunks[2][799], 3_999.0);
            assert_eq!(chunks[2][800], 0.0);
            acc.push(&[], |c| chunks.push(c.to_vec()));
            assert_eq!(chunks.len(), 3);
        }

        #[test]
        fn audio_line_base64_f32le_round_trip() {
            let chunk: Vec<f32> = (0..CHUNK_SAMPLES).map(|i| i as f32 / 800.0 - 1.0).collect();
            let line = audio_line("system", &chunk).to_string();
            let v: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(v["type"], "audio");
            assert_eq!(v["source"], "system");
            assert!(v["ts"].as_f64().unwrap() > 1.7e9);
            assert!((v["rms"].as_f64().unwrap() - rms(&chunk) as f64).abs() < 1e-6);
            let bytes = BASE64_STANDARD
                .decode(v["pcm_b64"].as_str().unwrap())
                .unwrap();
            assert_eq!(bytes.len(), CHUNK_SAMPLES * 4);
            let back: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(back, chunk);
            // 1.0f32 = 00 00 80 3F little-endian, standard alphabet, padded.
            assert_eq!(encode_pcm(&[1.0]), "AACAPw==");
        }

        #[test]
        fn rms_of_known_signals() {
            assert_eq!(rms(&[]), 0.0);
            assert_eq!(rms(&[0.0; CHUNK_SAMPLES]), 0.0);
            assert!((rms(&[0.5, -0.5, 0.5, -0.5]) - 0.5).abs() < 1e-7);
            assert!((rms(&[0.3, 0.4]) - 0.125f32.sqrt()).abs() < 1e-7);
        }

        #[test]
        fn silence_padding_arithmetic() {
            let ms = Duration::from_millis;
            let mut delivered = 0;
            assert_eq!(silence_padding(ms(99), &mut delivered), 0);
            assert_eq!(silence_padding(ms(100), &mut delivered), 1_600);
            assert_eq!(silence_padding(ms(250), &mut delivered), 2_400);
            assert_eq!(silence_padding(ms(250), &mut delivered), 0);
            assert_eq!(silence_padding(ms(200), &mut delivered), 0);
            assert_eq!(delivered, 4_000);

            // With jitter, the delivered count should stay within 100 ms of the clock.
            let mut delivered = 0;
            for tick in 1..=100u64 {
                let elapsed = ms(tick * 100 + tick % 7);
                silence_padding(elapsed, &mut delivered);
                let due = elapsed.as_millis() as u64 * 16;
                assert!(delivered <= due && due - delivered < 1_600, "tick {tick}");
            }
        }

        #[test]
        fn silence_padding_caps_catch_up_and_skips_the_rest() {
            let ms = Duration::from_millis;
            let mut delivered = 1_600;
            assert_eq!(silence_padding(ms(5_000), &mut delivered), 16_000);
            assert_eq!(delivered, 80_000);
            assert_eq!(silence_padding(ms(5_099), &mut delivered), 0);
            assert_eq!(silence_padding(ms(5_100), &mut delivered), 1_600);
        }

        #[test]
        fn resampler_output_length() {
            for rate in [8_000u32, 44_100, 48_000] {
                let mut r = Resampler::new(rate);
                let mut out = Vec::new();
                let mut input = vec![0.25f32; rate as usize]; // one second
                for part in input.chunks_mut(441) {
                    r.process(part, &mut out); // packet size unrelated to the ratio
                }
                assert_eq!(out.len(), 16_000, "{rate} Hz");
                // Allow the filter history to fill before checking its DC gain.
                assert!(
                    out[64..].iter().all(|&s| (s - 0.25).abs() < 1e-6),
                    "{rate} Hz: DC stays DC"
                );
            }
        }

        #[test]
        fn resampler_preserves_speech_band_and_attenuates_aliases() {
            let gain_db = |hz: f64| {
                let mut input: Vec<f32> = (0..48_000)
                    .map(|i| (TAU * hz * f64::from(i) / 48_000.0).sin() as f32)
                    .collect();
                let input_rms = rms(&input);
                let mut resampler = Resampler::new(48_000);
                let mut out = Vec::new();
                for part in input.chunks_mut(47) {
                    resampler.process(part, &mut out);
                }
                20.0 * (rms(&out[64..]) / input_rms).log10()
            };
            let speech = gain_db(1_000.0);
            let alias = gain_db(12_000.0);
            assert!(speech.abs() <= 1.0, "1 kHz gain: {speech} dB");
            assert!(alias <= -30.0, "12 kHz gain: {alias} dB");
        }

        #[test]
        fn converter_replaces_non_finite_samples_before_downmixing() {
            let stereo: Vec<u8> = [f32::NAN, 0.5, f32::INFINITY, 0.5, f32::NEG_INFINITY, 0.5]
                .repeat(160)
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            let mut out = Vec::new();
            Converter::new(SampleFormat::F32, 2, SAMPLE_RATE).push(&stereo, &mut out);
            assert_eq!(out, vec![0.25; 480]);

            out.clear();
            Converter::new(SampleFormat::F32, 2, 48_000).push(&stereo, &mut out);
            assert_eq!(out.len(), 160);
            assert!(out.iter().all(|s| s.is_finite()));
        }

        #[test]
        fn converter_decodes_downmixes_and_clamps() {
            let mut out = Vec::new();
            let stereo: Vec<u8> = [16_384i16, -8_192]
                .repeat(10)
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            Converter::new(SampleFormat::I16, 2, 16_000).push(&stereo, &mut out);
            assert_eq!(out, vec![0.125; 10]); // (0.5 + -0.25) / 2

            out.clear();
            let loud: Vec<u8> = [2.0f32, -3.0, 0.5]
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            Converter::new(SampleFormat::F32, 1, 16_000).push(&loud, &mut out);
            assert_eq!(out, vec![1.0, -1.0, 0.5]);

            out.clear();
            let mix = vec![0u8; 4_800 * 2 * 4]; // 100 ms of 48 kHz stereo f32
            Converter::new(SampleFormat::F32, 2, 48_000).push(&mix, &mut out);
            assert_eq!(out.len(), CHUNK_SAMPLES);
        }
    }
}

/// WASAPI capture: one event-driven thread per source.
#[cfg(windows)]
mod win {
    use crate::emit;
    use crate::pcm::{
        audio_line, silence_padding, ChunkAccumulator, Converter, SampleFormat, SAMPLE_RATE,
    };
    use serde_json::{json, Value};
    use std::collections::VecDeque;
    use std::io::{self, BufRead};
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    use std::sync::mpsc::{self, Sender};
    use std::thread;
    use std::time::{Duration, Instant};
    use wasapi::{
        AudioCaptureClient, AudioClient, DeviceEnumerator, Direction, Handle, SampleType,
        StreamMode, WasapiError, WaveFormat,
    };

    type Res<T> = Result<T, WasapiError>;

    /// WASAPI buffer, in 100 ns units (200 ms). Generous so that a briefly blocked stdout write
    /// cannot overrun it; latency is unaffected because packets are read as soon as the event fires.
    const BUFFER_HNS: i64 = 2_000_000;
    /// Event wait timeout: bounds how late `stop` is noticed and paces the silence padding.
    const WAIT_MS: u32 = 100;

    const E_ACCESSDENIED: u32 = 0x8007_0005;
    /// HRESULT_FROM_WIN32(ERROR_NOT_FOUND): GetDefaultAudioEndpoint when there is no such device.
    const E_NOTFOUND: u32 = 0x8007_0490;
    const AUDCLNT_E_DEVICE_INVALIDATED: u32 = 0x8889_0004;

    static STOP: AtomicBool = AtomicBool::new(false);
    /// Indexed by `Source as usize`.
    static MUTED: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];

    #[derive(Clone, Copy)]
    enum Source {
        System = 0,
        Microphone = 1,
    }

    enum Event {
        Started(Source, bool),
        Stop,
    }

    impl Source {
        fn name(self) -> &'static str {
            match self {
                Source::System => "system",
                Source::Microphone => "microphone",
            }
        }
    }

    fn emit_error(source: Source, message: String) {
        emit(&json!({"type": "error", "source": source.name(), "message": message}));
    }

    fn hresult(e: &WasapiError) -> u32 {
        match e {
            WasapiError::Windows(w) => w.code().0 as u32,
            _ => 0,
        }
    }

    /// An initialised capture stream plus the converter for the format WASAPI accepted.
    struct Stream {
        capture: AudioCaptureClient,
        client: AudioClient,
        event: Handle,
        conv: Converter,
    }

    /// Activates a fresh client in shared event mode. `fmt: None` uses the endpoint mix format.
    fn init(
        activate: &dyn Fn() -> Res<AudioClient>,
        fmt: Option<WaveFormat>,
        mode: StreamMode,
    ) -> Result<Stream, (&'static str, WasapiError)> {
        let mut client = activate().map_err(|e| ("activate", e))?;
        let fmt = match fmt {
            Some(fmt) => fmt,
            None => client.get_mixformat().map_err(|e| ("get mix format", e))?,
        };
        let subformat = fmt.get_subformat().map_err(|e| ("get sample format", e))?;
        let sample = match (subformat, fmt.get_bitspersample()) {
            (SampleType::Float, 32) => SampleFormat::F32,
            (SampleType::Int, 16) => SampleFormat::I16,
            (SampleType::Int, 32) => SampleFormat::I32,
            _ => return Err(("get sample format", WasapiError::UnsupportedFormat)),
        };
        client
            .initialize_client(&fmt, Direction::Capture, mode, None)
            .map_err(|e| ("initialize", e))?;
        let event = client
            .set_get_eventhandle()
            .map_err(|e| ("set event handle", e))?;
        let capture = client
            .get_audiocaptureclient()
            .map_err(|e| ("get capture client", e))?;
        let conv = Converter::new(sample, fmt.get_nchannels().into(), fmt.get_samplespersec());
        Ok(Stream {
            capture,
            client,
            event,
            conv,
        })
    }

    /// Capture attempts include Start. The microphone permission probe only initializes.
    fn open_attempt(
        label: &str,
        activate: &dyn Fn() -> Res<AudioClient>,
        fmt: Option<WaveFormat>,
        autoconvert: bool,
        buffer_duration_hns: i64,
        start: bool,
    ) -> Res<Stream> {
        let mode = StreamMode::EventsShared {
            autoconvert,
            buffer_duration_hns,
        };
        init(activate, fmt, mode)
            .and_then(|stream| {
                if start {
                    stream.client.start_stream().map_err(|e| ("start", e))?;
                }
                Ok(stream)
            })
            .map_err(|(stage, e)| {
                if start {
                    eprintln!("{label} (autoconvert={autoconvert}, buffer={buffer_duration_hns} hns) {stage} failed: {e}");
                }
                e
            })
    }

    /// Try 16 kHz mono f32 first, then the endpoint mix format with software conversion.
    fn open_endpoint(direction: Direction, label: &str, start: bool) -> Res<Stream> {
        let device = DeviceEnumerator::new()
            .and_then(|en| en.get_default_device(direction))
            .inspect_err(|e| {
                if start {
                    eprintln!("{label} get default endpoint failed: {e}");
                }
            })?;
        let activate = || device.get_iaudioclient();
        let wanted = WaveFormat::new(32, 32, SampleType::Float, SAMPLE_RATE as usize, 1, None);
        open_attempt(label, &activate, Some(wanted), true, BUFFER_HNS, start)
            .or_else(|_| open_attempt(label, &activate, None, false, BUFFER_HNS, start))
    }

    /// System audio = everything the user hears except `exclude_pid` and its child processes (the
    /// app's WebView2 plays our own Vietnamese TTS dubbing). Returns the stream and the status note.
    fn open_system(exclude_pid: u32) -> Res<(Stream, Option<&'static str>)> {
        // Process loopback (Windows 10 build 20348+ / Windows 11): ActivateAudioInterfaceAsync on
        // VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK with AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK.
        // In wasapi 0.27.0, `include_tree == false` selects PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE
        // (checked in its src/api.rs, despite its doc comment). That is the mode we need.
        let process_loopback = || AudioClient::new_application_loopback_client(exclude_pid, false);
        for (rate, channels, autoconvert, buffer) in [
            (SAMPLE_RATE as usize, 1, true, BUFFER_HNS),
            // The 48 kHz autoconvert attempt matches wasapi's record_application example.
            (48_000, 2, true, 0),
            (48_000, 2, false, 0),
        ] {
            let fmt = WaveFormat::new(32, 32, SampleType::Float, rate, channels, None);
            if let Ok(stream) = open_attempt(
                "process loopback",
                &process_loopback,
                Some(fmt),
                autoconvert,
                buffer,
                true,
            ) {
                return Ok((stream, None));
            }
        }
        // Classic loopback of the default render endpoint also hears our own dubbing.
        open_endpoint(Direction::Render, "classic loopback", true)
            .map(|stream| (stream, Some("loopback-no-exclusion")))
    }

    fn open_error(source: Source, e: &WasapiError) -> String {
        match (source, hresult(e)) {
            (Source::Microphone, E_ACCESSDENIED) => "Windows đang chặn quyền Micro: mở Cài đặt > Quyền riêng tư & bảo mật > Micro và bật 'Cho phép ứng dụng máy tính truy cập micro'.".into(),
            (Source::Microphone, E_NOTFOUND | AUDCLNT_E_DEVICE_INVALIDATED) => "Không tìm thấy thiết bị Micro đầu vào.".into(),
            (Source::System, E_NOTFOUND | AUDCLNT_E_DEVICE_INVALIDATED) => "Không tìm thấy thiết bị phát âm thanh (loa/tai nghe) để thu âm thanh hệ thống.".into(),
            (Source::Microphone, _) => format!("Không mở được Micro: {e}"),
            (Source::System, _) => format!("Không thu được âm thanh hệ thống: {e}"),
        }
    }

    fn capture_thread(source: Source, exclude_pid: u32, events: Sender<Event>) {
        if STOP.load(Relaxed) {
            return;
        }
        // COM on every thread that touches WASAPI. MTA, so process-loopback activation completes
        // on a worker thread without needing a message pump here.
        if let Err(e) = wasapi::initialize_mta().ok() {
            emit_error(source, format!("Không khởi tạo được COM: {e}"));
            let _ = events.send(Event::Started(source, false));
            return;
        }
        capture(source, exclude_pid, &events); // every COM object is dropped inside, before CoUninitialize
        wasapi::deinitialize();
    }

    fn capture(source: Source, exclude_pid: u32, events: &Sender<Event>) {
        let opened = match source {
            Source::System => open_system(exclude_pid),
            Source::Microphone => {
                open_endpoint(Direction::Capture, "microphone", true).map(|stream| (stream, None))
            }
        };
        if STOP.load(Relaxed) {
            if let Ok((stream, _)) = opened {
                let _ = stream.client.stop_stream();
            }
            return;
        }
        let (mut stream, note) = match opened {
            Ok(ok) => ok,
            Err(e) => {
                emit_error(source, open_error(source, &e));
                let _ = events.send(Event::Started(source, false));
                return;
            }
        };
        let mut status = json!({"type": "status", "source": source.name(), "state": "capturing"});
        if let Some(note) = note {
            status["note"] = note.into();
        }
        emit(&status);
        let _ = events.send(Event::Started(source, true));

        if let Err(e) = pump(&mut stream, source) {
            let what = match source {
                Source::System => "âm thanh hệ thống",
                Source::Microphone => "Micro",
            };
            emit_error(source, format!("Thu {what} bị gián đoạn: {e}"));
        }
        let _ = stream.client.stop_stream();
    }

    /// Event-driven read loop until `stop`; returns early on a device error.
    fn pump(s: &mut Stream, source: Source) -> Res<()> {
        let pad = matches!(source, Source::System);
        let mut acc = ChunkAccumulator::default();
        let mut raw = VecDeque::new();
        let mut mono = Vec::new();
        let start = Instant::now();
        let mut delivered = 0; // 16 kHz samples accounted for, including muted or skipped samples
        while !STOP.load(Relaxed) {
            // A timeout is normal (loopback is silent while nothing plays): still poll, then pad.
            let _ = s.event.wait_for_event(WAIT_MS);
            if STOP.load(Relaxed) {
                break;
            }
            while s.capture.get_next_packet_size()?.unwrap_or(0) > 0 {
                // wasapi zero-fills packets flagged AUDCLNT_BUFFERFLAGS_SILENT.
                s.capture.read_from_device_to_deque(&mut raw)?;
            }
            s.conv.push(raw.make_contiguous(), &mut mono);
            raw.clear();
            if pad && mono.is_empty() {
                mono.resize(silence_padding(start.elapsed(), &mut delivered), 0.0);
            } else {
                delivered += mono.len() as u64;
            }
            if !MUTED[source as usize].load(Relaxed) {
                acc.push(&mono, |chunk| emit(&audio_line(source.name(), chunk)));
            }
            mono.clear();
        }
        Ok(())
    }

    fn check() {
        let initialized = wasapi::initialize_mta().is_ok();
        // Activate + Initialize the default capture endpoint, then release it. With "Let desktop
        // apps access your microphone" off this fails with E_ACCESSDENIED, and with no microphone
        // with E_NOTFOUND. Any failure means capture would fail too, so it reports "denied".
        let microphone = match open_endpoint(Direction::Capture, "microphone", false) {
            Ok(_) => "granted",
            Err(e) => {
                eprintln!("microphone check failed: {e}");
                "denied"
            }
        };
        // Loopback needs no permission on Windows, only a render endpoint.
        let render =
            DeviceEnumerator::new().and_then(|en| en.get_default_device(Direction::Render));
        let system_audio = if render.is_ok() { "granted" } else { "denied" };
        drop(render);
        if initialized {
            wasapi::deinitialize();
        }
        emit(
            &json!({"type": "permissions", "microphone": microphone, "system_audio": system_audio}),
        );
    }

    pub fn run() -> i32 {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let has = |flag: &str| args.iter().any(|a| a == flag);
        if has("--check") {
            check();
            return 0;
        }
        // Process loopback excludes exactly one process tree. The backend passes the app's PID,
        // whose tree holds its WebView2 processes and this helper; without the flag, exclude ourselves.
        let exclude_pid = args
            .iter()
            .position(|a| a == "--exclude-pid")
            .and_then(|i| args.get(i + 1)?.parse().ok())
            .unwrap_or_else(std::process::id);

        let (events_tx, events) = mpsc::channel();
        let stdin_tx = events_tx.clone();
        thread::spawn(move || read_stdin(stdin_tx));

        let mut threads = Vec::new();
        for (flag, source) in [("--system", Source::System), ("--mic", Source::Microphone)] {
            if has(flag) {
                let tx = events_tx.clone();
                threads.push(thread::spawn(move || {
                    capture_thread(source, exclude_pid, tx)
                }));
            }
        }
        drop(events_tx);

        // Wait for startup results, a stop from stdin, or the startup deadline.
        let mut running = [false; 2];
        let deadline = Instant::now() + Duration::from_secs(10);
        for _ in 0..threads.len() {
            match events.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(Event::Started(source, ok)) => running[source as usize] = ok,
                Ok(Event::Stop) | Err(_) => break,
            }
        }
        if !STOP.load(Relaxed) {
            emit(&json!({"type": "ready", "system": running[0], "microphone": running[1]}));
            while let Ok(Event::Started(..)) = events.recv() {}
        }

        // A thread stuck inside WASAPI must not keep the process alive.
        let deadline = Instant::now() + Duration::from_secs(1);
        while threads.iter().any(|t| !t.is_finished()) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            thread::sleep(remaining.min(Duration::from_millis(10)));
        }
        emit(&json!({"type": "stopped"}));
        0
    }

    fn read_stdin(events: Sender<Event>) {
        for line in io::stdin().lock().lines() {
            let line = match line {
                Ok(line) => line,
                Err(e) if e.kind() == io::ErrorKind::InvalidData => continue, // not UTF-8: ignore it
                Err(_) => break,
            };
            let cmd = serde_json::from_str::<Value>(line.trim())
                .ok()
                .and_then(|v| v.get("cmd")?.as_str().map(str::to_owned));
            match cmd.as_deref() {
                Some("mute_mic") => MUTED[Source::Microphone as usize].store(true, Relaxed),
                Some("unmute_mic") => MUTED[Source::Microphone as usize].store(false, Relaxed),
                Some("mute_system") => MUTED[Source::System as usize].store(true, Relaxed),
                Some("unmute_system") => MUTED[Source::System as usize].store(false, Relaxed),
                Some("stop") => break,
                _ => {}
            }
        }

        STOP.store(true, Relaxed);
        let _ = events.send(Event::Stop);
    }
}
