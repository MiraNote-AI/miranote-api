// Apple Vision instance-mask bench.
//
// Answers one question about the 17 images in test_input/: does Vision's own
// instance decomposition contain the subject each prompt asks for? SAM takes a
// box and cuts out whatever is inside it; Vision only hands back the foreground
// instances it found on its own, so a detector box can pick from that menu but
// cannot change it. If the menu never separates "the boy on the left", the
// Gemini+Vision pipeline cannot either -- no amount of prompting fixes it.
//
// Runs two requests per image:
//   GenerateForegroundInstanceMaskRequest  -- class-agnostic, saliency-driven
//   GeneratePersonInstanceMaskRequest      -- people only, separates individuals
// The second exists because 8 of the 17 cases are people and the hard cases
// ("the girl on the right") are exactly where generic saliency tends to merge.
//
//   swiftc -O -parse-as-library vision_bench.swift -o vision_bench
//   ./vision_bench --input test_input --out test_output/vision

import Foundation
import Vision
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// MARK: - Inputs

struct BenchCase: Decodable {
    let image: String
    let prompt: String
    let prefix: String?
}

struct InstanceInfo: Codable {
    let index: Int
    let areaFraction: Double
    let bbox: [Int]          // x0, y0, x1, y1 in pixels of the normalized image
}

struct RequestResult: Codable {
    /// Time for `perform` alone -- the model run.
    var ms: Double = 0
    /// Time to upscale the instance masks to full resolution. Separate because
    /// `perform` returns a low-res observation and defers the upscale, so
    /// quoting `ms` alone would understate what a real cutout costs.
    var maskMs: Double = 0
    var instances: [InstanceInfo] = []
    var error: String?
}

struct ImageResult: Codable {
    let prefix: String
    let image: String
    let prompt: String
    let width: Int
    let height: Int
    var fg = RequestResult()
    var person = RequestResult()
}

// MARK: - Image loading

/// Load an image with EXIF orientation applied and the long side capped.
///
/// Orientation is applied here rather than handed to Vision so that every
/// artifact downstream -- the masks, the cutouts, and the source PNG the review
/// sheets composite against -- lives in one coordinate space. `5.jpeg` carries
/// EXIF orientation 6; without this its mask comes back rotated 90 degrees
/// relative to how PIL and every other tool renders it.
///
/// The cap matches `_shrink_for_model`'s 1536 in main.py so the timings here are
/// comparable to the 8.77 s the current DINO+SAM pipeline takes on the same input.
func loadNormalized(url: URL, maxSide: Int) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }

    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    let rawOrientation = props?[kCGImagePropertyOrientation] as? UInt32 ?? 1
    let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) ?? .up

    // CGImageSourceCreateImageAtIndex ignores EXIF orientation, so applying it
    // once here cannot double-apply the way CIImage(contentsOf:) might.
    var ci = CIImage(cgImage: cg).oriented(orientation)
    let side = max(ci.extent.width, ci.extent.height)
    if side > CGFloat(maxSide) {
        let scale = CGFloat(maxSide) / side
        ci = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
    return ciContext.createCGImage(ci, from: ci.extent)
}

let ciContext = CIContext(options: [.useSoftwareRenderer: false])

// MARK: - CVPixelBuffer -> CGImage

/// Single-channel mask buffer to an 8-bit grayscale CGImage.
///
/// generateScaledMask documents kCVPixelFormatType_OneComponent32Float for the
/// low-res variant and leaves the scaled one unspecified, so both float and
/// 8-bit are handled; anything else falls through to the CoreImage path.
func grayCGImage(from buffer: CVPixelBuffer) -> CGImage? {
    let format = CVPixelBufferGetPixelFormatType(buffer)
    guard format == kCVPixelFormatType_OneComponent32Float
            || format == kCVPixelFormatType_OneComponent8 else { return nil }

    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

    var pixels = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
        let row = base.advanced(by: y * rowBytes)
        if format == kCVPixelFormatType_OneComponent32Float {
            let floats = row.assumingMemoryBound(to: Float.self)
            for x in 0..<width {
                pixels[y * width + x] = UInt8(max(0, min(255, floats[x] * 255)))
            }
        } else {
            let bytes = row.assumingMemoryBound(to: UInt8.self)
            for x in 0..<width { pixels[y * width + x] = bytes[x] }
        }
    }

    guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
    return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                   bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: false,
                   intent: .defaultIntent)
}

/// Any pixel buffer to a CGImage, preserving alpha. Used for the RGBA cutouts
/// generateMaskedImage returns, and as the fallback for unexpected mask formats.
func colorCGImage(from buffer: CVPixelBuffer) -> CGImage? {
    let ci = CIImage(cvPixelBuffer: buffer)
    return ciContext.createCGImage(ci, from: ci.extent,
                                   format: .RGBA8,
                                   colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
}

func writePNG(_ image: CGImage, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                            withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw BenchError.message("cannot create PNG destination at \(url.path)")
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw BenchError.message("cannot write PNG at \(url.path)")
    }
}

enum BenchError: Error, CustomStringConvertible {
    case message(String)
    var description: String { if case .message(let m) = self { return m }; return "error" }
}

/// Tight bounding box and coverage of a grayscale mask, so the review sheet can
/// say how much of the frame each instance takes without re-reading the PNGs.
func maskStats(_ image: CGImage) -> (bbox: [Int], areaFraction: Double)? {
    let width = image.width, height = image.height
    var pixels = [UInt8](repeating: 0, count: width * height)
    guard let ctx = CGContext(data: &pixels, width: width, height: height,
                              bitsPerComponent: 8, bytesPerRow: width,
                              space: CGColorSpaceCreateDeviceGray(),
                              bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    var minX = width, minY = height, maxX = -1, maxY = -1, count = 0
    for y in 0..<height {
        for x in 0..<width where pixels[y * width + x] > 127 {
            count += 1
            if x < minX { minX = x }; if x > maxX { maxX = x }
            if y < minY { minY = y }; if y > maxY { maxY = y }
        }
    }
    guard count > 0 else { return ([0, 0, 0, 0], 0) }
    return ([minX, minY, maxX + 1, maxY + 1], Double(count) / Double(width * height))
}

// MARK: - One request over one image

/// Run one instance-mask request and write every instance's mask and cutout.
///
/// `generateMaskedImage` hands back the RGBA cutout directly, so the alpha
/// compositing that would otherwise happen in Python is free here and the
/// review sheets show exactly what Vision produced.
func runInstanceRequest<R: ImageProcessingRequest>(
    _ request: R,
    handler: ImageRequestHandler,
    outDir: URL,
    label: String
) async -> RequestResult where R.Result == InstanceMaskObservation? {
    var result = RequestResult()

    let start = DispatchTime.now().uptimeNanoseconds
    let observation: InstanceMaskObservation?
    do {
        observation = try await handler.perform(request)
    } catch {
        result.ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        result.error = "\(error)"
        return result
    }
    result.ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6

    guard let observation, !observation.allInstances.isEmpty else { return result }

    for index in observation.allInstances {
        do {
            let single = IndexSet(integer: index)
            let maskStart = DispatchTime.now().uptimeNanoseconds
            let maskBuffer = try observation.generateScaledMask(
                for: single, scaledToImageFrom: handler)
            result.maskMs += Double(DispatchTime.now().uptimeNanoseconds - maskStart) / 1e6
            guard let mask = grayCGImage(from: maskBuffer) ?? colorCGImage(from: maskBuffer) else {
                result.error = "instance \(index): unsupported mask pixel format"
                continue
            }
            try writePNG(mask, to: outDir.appendingPathComponent("inst_\(index)_mask.png"))

            let cutoutBuffer = try observation.generateMaskedImage(
                for: single, imageFrom: handler, croppedToInstancesExtent: false)
            if let cutout = colorCGImage(from: cutoutBuffer) {
                try writePNG(cutout, to: outDir.appendingPathComponent("inst_\(index).png"))
            }

            let stats = maskStats(mask)
            result.instances.append(InstanceInfo(
                index: index,
                areaFraction: (stats?.areaFraction).map { ($0 * 10000).rounded() / 10000 } ?? 0,
                bbox: stats?.bbox ?? []))
        } catch {
            result.error = "instance \(index): \(error)"
        }
    }

    // The union doubles as the rembg replacement: if it is clean, Vision can
    // stand in for the 12.55 s background-removal stage regardless of whether
    // the per-instance split is good enough to also replace SAM.
    if observation.allInstances.count > 1 {
        if let allBuffer = try? observation.generateMaskedImage(
            for: observation.allInstances, imageFrom: handler, croppedToInstancesExtent: false),
           let all = colorCGImage(from: allBuffer) {
            try? writePNG(all, to: outDir.appendingPathComponent("all.png"))
        }
    }

    _ = label
    return result
}

// MARK: - Serve mode

/// Long-lived matte helper: one request per stdin line, `<in_path>\t<out_path>`.
///
/// A process per request would pay the ~1.4 s model load every time, which turns
/// a 0.03 s matte into a 1.5 s one -- still eight times faster than rembg, but
/// fifty times slower than it needs to be. Staying resident pays that once.
///
/// Every reply is one JSON line, so a failure is a value the caller reads rather
/// than a hang: main.py's mode surfaces it instead of quietly falling back.
func serve(maxSide: Int) async {
    setvbuf(stdout, nil, _IOLBF, 0)

    // Warm the model before announcing readiness, so the first real request is
    // not the one that pays for the load.
    if let seed = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8,
                            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?
        .makeImage() {
        _ = try? await ImageRequestHandler(seed)
            .perform(GenerateForegroundInstanceMaskRequest())
    }
    print(#"{"ready":true}"#)

    while let line = readLine(strippingNewline: true) {
        if line.isEmpty { continue }
        let parts = line.components(separatedBy: "\t")
        guard parts.count == 2 else {
            print(#"{"ok":false,"error":"expected <in>\t<out>"}"#)
            continue
        }
        let started = DispatchTime.now().uptimeNanoseconds
        do {
            guard let image = loadNormalized(url: URL(fileURLWithPath: parts[0]),
                                             maxSide: maxSide) else {
                throw BenchError.message("cannot load \(parts[0])")
            }
            let handler = ImageRequestHandler(image)
            guard let observation = try await handler.perform(
                    GenerateForegroundInstanceMaskRequest()),
                  !observation.allInstances.isEmpty else {
                throw BenchError.message("no foreground instances")
            }
            // Union of every instance: this stands in for rembg, which returns
            // one whole-foreground matte rather than a chosen subject.
            let buffer = try observation.generateMaskedImage(
                for: observation.allInstances, imageFrom: handler,
                croppedToInstancesExtent: false)
            guard let cutout = colorCGImage(from: buffer) else {
                throw BenchError.message("unsupported cutout pixel format")
            }
            try writePNG(cutout, to: URL(fileURLWithPath: parts[1]))
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            print(String(format: #"{"ok":true,"ms":%.1f,"instances":%d}"#,
                         ms, observation.allInstances.count))
        } catch {
            let escaped = "\(error)".replacingOccurrences(of: "\"", with: "'")
                .replacingOccurrences(of: "\n", with: " ")
            print("{\"ok\":false,\"error\":\"\(escaped)\"}")
        }
    }
}

// MARK: - Entry point

@main
struct VisionBench {
    static func main() async {
        var inputDir = "test_input"
        var outDir = "test_output/vision"
        var casesPath = "test_input/cases.json"
        var maxSide = 1536
        var warmup = true
        var serveMode = false

        var args = Array(CommandLine.arguments.dropFirst())
        while let flag = args.first {
            args.removeFirst()
            switch flag {
            case "--input":  inputDir = args.removeFirst()
            case "--out":    outDir = args.removeFirst()
            case "--cases":  casesPath = args.removeFirst()
            case "--max-side": maxSide = Int(args.removeFirst()) ?? 1536
            case "--no-warmup": warmup = false
            case "--serve":  serveMode = true
            default:
                FileHandle.standardError.write("unknown flag \(flag)\n".data(using: .utf8)!)
                exit(2)
            }
        }

        if serveMode {
            await serve(maxSide: maxSide)
            return
        }

        guard let casesData = FileManager.default.contents(atPath: casesPath),
              let cases = try? JSONDecoder().decode([BenchCase].self, from: casesData) else {
            FileHandle.standardError.write("cannot read cases at \(casesPath)\n".data(using: .utf8)!)
            exit(1)
        }
        _ = inputDir  // case paths in cases.json are already repo-relative

        let out = URL(fileURLWithPath: outDir)
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        // Vision loads its models on the first request; without a discarded
        // warmup that one-off cost lands entirely on the first case and
        // misreports it as the slowest, the same reason bench_cutout.py warms up.
        if warmup, let first = cases.first,
           let image = loadNormalized(url: URL(fileURLWithPath: first.image), maxSide: maxSide) {
            let handler = ImageRequestHandler(image)
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try? await handler.perform(GenerateForegroundInstanceMaskRequest())
            _ = try? await handler.perform(GeneratePersonInstanceMaskRequest())
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            print(String(format: "warmup (discarded, includes model load): %.0f ms", ms))
        }

        var results: [ImageResult] = []
        for bench in cases {
            let prefix = bench.prefix ?? URL(fileURLWithPath: bench.image)
                .deletingPathExtension().lastPathComponent
            guard let image = loadNormalized(url: URL(fileURLWithPath: bench.image),
                                             maxSide: maxSide) else {
                print("\(prefix): cannot load \(bench.image)")
                continue
            }

            try? writePNG(image, to: out.appendingPathComponent("source/\(prefix).png"))
            let handler = ImageRequestHandler(image)

            var row = ImageResult(prefix: prefix, image: bench.image, prompt: bench.prompt,
                                  width: image.width, height: image.height)
            row.fg = await runInstanceRequest(
                GenerateForegroundInstanceMaskRequest(), handler: handler,
                outDir: out.appendingPathComponent("fg/\(prefix)"), label: "fg")
            row.person = await runInstanceRequest(
                GeneratePersonInstanceMaskRequest(), handler: handler,
                outDir: out.appendingPathComponent("person/\(prefix)"), label: "person")
            results.append(row)

            print(String(format: "%-26s %4dx%-4d  fg %2d inst %5.0f+%-5.0f ms   person %2d inst %5.0f+%-5.0f ms",
                         (prefix as NSString).utf8String!, image.width, image.height,
                         row.fg.instances.count, row.fg.ms, row.fg.maskMs,
                         row.person.instances.count, row.person.ms, row.person.maskMs))
            for note in [row.fg.error, row.person.error].compactMap({ $0 }) {
                print("    ! \(note)")
            }
        }

        let encoder = JSONEncoder()
        let lines = results.compactMap { try? encoder.encode($0) }
            .compactMap { String(data: $0, encoding: .utf8) }
        try? lines.joined(separator: "\n").write(
            to: out.appendingPathComponent("vision_bench.jsonl"),
            atomically: true, encoding: .utf8)
        print("\nWrote \(results.count) rows to \(outDir)/vision_bench.jsonl")
    }
}
