import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

final class OutputEmitter: @unchecked Sendable {
    private let lock = NSLock()
    func send(_ obj: [String: Any]) {
        lock.lock()
        defer { lock.unlock() }
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
           let line = String(data: data, encoding: .utf8),
           let outData = (line + "\n").data(using: .utf8) {
            FileHandle.standardOutput.write(outData)
        }
    }
}

final class PCMChunkAccumulator: @unchecked Sendable {
    static let targetSampleRate: Double = 16000.0
    static let chunkSamples: Int = 1600 // 100 ms @ 16 kHz

    private let source: String
    private let emitter: OutputEmitter
    private var buffer: [Float] = []
    private let lock = NSLock()
    private var _muted: Bool = false

    var muted: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _muted }
        set { lock.lock(); _muted = newValue; lock.unlock() }
    }

    init(source: String, emitter: OutputEmitter) {
        self.source = source
        self.emitter = emitter
        self.buffer.reserveCapacity(Self.chunkSamples * 2)
    }

    func append(_ samples: UnsafeBufferPointer<Float>) {
        if muted || samples.isEmpty { return }
        lock.lock()
        buffer.append(contentsOf: samples)
        while buffer.count >= Self.chunkSamples {
            let slice = Array(buffer.prefix(Self.chunkSamples))
            buffer.removeFirst(Self.chunkSamples)
            lock.unlock()
            emitSlice(slice)
            lock.lock()
        }
        lock.unlock()
    }

    private func emitSlice(_ slice: [Float]) {
        var rmsSum: Float = 0
        for s in slice { rmsSum += s * s }
        let rms = sqrt(rmsSum / Float(slice.count))
        let byteData = slice.withUnsafeBufferPointer { Data(buffer: $0) }
        emitter.send([
            "type": "audio",
            "source": source,
            "ts": Date().timeIntervalSince1970,
            "rms": rms,
            "pcm_b64": byteData.base64EncodedString()
        ])
    }
}

final class SystemAudioDelegate: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    let acc: PCMChunkAccumulator
    private let emitter: OutputEmitter

    init(emitter: OutputEmitter) {
        self.emitter = emitter
        self.acc = PCMChunkAccumulator(source: "system", emitter: emitter)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .audio, sampleBuffer.isValid else { return }
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }
        let asbd = asbdPtr.pointee

        var blockBuffer: CMBlockBuffer?
        var audioBufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: 0, mData: nil)
        )
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )
        guard status == noErr else { return }

        let buf = audioBufferList.mBuffers
        guard let rawPtr = buf.mData, buf.mDataByteSize > 0 else { return }

        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        if isFloat && asbd.mBitsPerChannel == 32 {
            let count = Int(buf.mDataByteSize) / MemoryLayout<Float>.size
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let channels = max(1, Int(asbd.mChannelsPerFrame))
            if channels == 1 {
                acc.append(UnsafeBufferPointer(start: floatPtr, count: count))
            } else {
                let frames = count / channels
                var mono = [Float](repeating: 0, count: frames)
                for f in 0..<frames {
                    var sum: Float = 0
                    for c in 0..<channels { sum += floatPtr[f * channels + c] }
                    mono[f] = sum / Float(channels)
                }
                mono.withUnsafeBufferPointer { acc.append($0) }
            }
        } else if !isFloat && asbd.mBitsPerChannel == 16 {
            let count = Int(buf.mDataByteSize) / MemoryLayout<Int16>.size
            let intPtr = rawPtr.assumingMemoryBound(to: Int16.self)
            var floats = [Float](repeating: 0, count: count)
            for i in 0..<count { floats[i] = Float(intPtr[i]) / 32768.0 }
            floats.withUnsafeBufferPointer { acc.append($0) }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        emitter.send(["type": "error", "source": "system", "message": error.localizedDescription])
    }
}

final class MicrophoneCapturer: @unchecked Sendable {
    private let engine = AVAudioEngine()
    let acc: PCMChunkAccumulator
    private var converter: AVAudioConverter?

    init(emitter: OutputEmitter) {
        self.acc = PCMChunkAccumulator(source: "microphone", emitter: emitter)
    }

    func start() throws {
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0 && hwFormat.channelCount > 0 else {
            throw NSError(domain: "VinaNoteAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "Không tìm thấy thiết bị Microphone đầu vào."])
        }
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: PCMChunkAccumulator.targetSampleRate, channels: 1, interleaved: false) else {
            throw NSError(domain: "VinaNoteAudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "Không tạo được định dạng 16kHz PCM."])
        }
        guard let conv = AVAudioConverter(from: hwFormat, to: outFormat) else {
            throw NSError(domain: "VinaNoteAudio", code: 3, userInfo: [NSLocalizedDescriptionKey: "Không khởi tạo được AVAudioConverter."])
        }
        self.converter = conv

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: hwFormat) { [weak self] pcmBuf, _ in
            guard let self = self, let conv = self.converter else { return }
            let ratio = PCMChunkAccumulator.targetSampleRate / hwFormat.sampleRate
            let outCapacity = AVAudioFrameCount(Double(pcmBuf.frameLength) * ratio + 64)
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else { return }

            var consumed = false
            var convErr: NSError?
            conv.convert(to: outBuf, error: &convErr) { _, outStatus in
                if consumed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                consumed = true
                outStatus.pointee = .haveData
                return pcmBuf
            }
            if convErr == nil, outBuf.frameLength > 0, let ch = outBuf.floatChannelData?[0] {
                self.acc.append(UnsafeBufferPointer(start: ch, count: Int(outBuf.frameLength)))
            }
        }
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

@main
struct AudioCapCLI {
    static func main() async {
        let emitter = OutputEmitter()
        let args = CommandLine.arguments
        let enableSystem = args.contains("--system")
        let enableMic = args.contains("--mic")
        let checkOnly = args.contains("--check")

        var excludePids: Set<pid_t> = [ProcessInfo.processInfo.processIdentifier]
        if let idx = args.firstIndex(of: "--exclude-pid"), idx + 1 < args.count, let pid = Int32(args[idx + 1]) {
            excludePids.insert(pid)
        }

        if checkOnly {
            let micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
            var screenOk = false
            do {
                _ = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                screenOk = true
            } catch {
                screenOk = false
            }
            emitter.send([
                "type": "permissions",
                "microphone": micStatus == .authorized ? "granted" : (micStatus == .notDetermined ? "prompt" : "denied"),
                "system_audio": screenOk ? "granted" : "denied"
            ])
            return
        }

        var scStream: SCStream?
        let sysDelegate = SystemAudioDelegate(emitter: emitter)
        let micCap = MicrophoneCapturer(emitter: emitter)
        var micRunning = false

        if enableSystem {
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
                guard let display = content.displays.first else {
                    emitter.send(["type": "error", "source": "system", "message": "Không tìm thấy màn hình để gắn bộ thu âm hệ thống."])
                    return
                }
                let excludedApps = content.applications.filter { excludePids.contains($0.processID) }
                let filter = SCContentFilter(display: display, excludingApplications: excludedApps, exceptingWindows: [])
                let cfg = SCStreamConfiguration()
                cfg.capturesAudio = true
                cfg.excludesCurrentProcessAudio = true
                cfg.sampleRate = Int(PCMChunkAccumulator.targetSampleRate)
                cfg.channelCount = 1
                cfg.width = 2
                cfg.height = 2
                cfg.minimumFrameInterval = CMTime(value: 10, timescale: 1)

                let stream = SCStream(filter: filter, configuration: cfg, delegate: sysDelegate)
                try stream.addStreamOutput(sysDelegate, type: .audio, sampleHandlerQueue: DispatchQueue(label: "vinanote.sysaudio", qos: .userInteractive))
                try await stream.startCapture()
                scStream = stream
                emitter.send(["type": "status", "source": "system", "state": "capturing"])
            } catch {
                emitter.send(["type": "error", "source": "system", "message": "Cần cấp quyền Ghi màn hình & Âm thanh hệ thống trong System Settings: \(error.localizedDescription)"])
            }
        }

        if enableMic {
            let granted = await AVCaptureDevice.requestAccess(for: .audio)
            if granted {
                do {
                    try micCap.start()
                    micRunning = true
                    emitter.send(["type": "status", "source": "microphone", "state": "capturing"])
                } catch {
                    emitter.send(["type": "error", "source": "microphone", "message": error.localizedDescription])
                }
            } else {
                emitter.send(["type": "error", "source": "microphone", "message": "Quyền Microphone bị từ chối trong System Settings."])
            }
        }

        emitter.send(["type": "ready", "system": scStream != nil, "microphone": micRunning])

        let stopSemaphore = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async {
            while let line = readLine() {
                guard let data = line.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let cmd = obj["cmd"] as? String else { continue }
                switch cmd {
                case "mute_mic": micCap.acc.muted = true
                case "unmute_mic": micCap.acc.muted = false
                case "mute_system": sysDelegate.acc.muted = true
                case "unmute_system": sysDelegate.acc.muted = false
                case "stop":
                    stopSemaphore.signal()
                    return
                default: break
                }
            }
            stopSemaphore.signal()
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                stopSemaphore.wait()
                cont.resume()
            }
        }

        if let s = scStream {
            try? await s.stopCapture()
        }
        if micRunning {
            micCap.stop()
        }
        emitter.send(["type": "stopped"])
    }
}
