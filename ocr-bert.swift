#!/usr/bin/env swift
//
// ocr-nlp-bert-gemini.swift
//
// Potok OCR PDF -> PDF przeszukiwalny (Wersja z pełną akceleracją Apple ML):
//
//  - [ML] Inteligentne kadrowanie skanów (VNDetectDocumentSegmentationRequest + VNDetectContoursRequest)
//  - [ML] Automatyczne prostowanie pochylenia strony przed OCR (VNDetectHorizonRequest)
//  - [ML] Rekonstrukcja ostrości małych liter sieciami neuronowymi (VNGenerateImageSuperResolutionRequest)
//  - [ML] Automatyczna detekcja języka i dynamiczny dobór słowników (NLLanguageRecognizer)
//  - [ML] Klasyfikacja stron-fotografii (VNClassifyImageRequest)
//  - [ML] Adaptacyjne powiększenie wg jakości skanu (VNCalculateImageAestheticsScoresRequest, macOS 15+)
//  - [ML] Ochrona nazw własnych przed korektą diakrytyków (NLTagger)
//  - [ML] Ulepszone rozstrzyganie korekty (NLEmbedding/NLContextualEmbedding + Symetryczna Matryca Konfuzji i Grafemów)
//  - Automatyczny fizyczny podział szerokich stron (rozkładówek) na osobne strony pionowe
//  - Dwuprzebiegowy OCR (tekst główny + drobny druk) z deduplikacją (IoU)
//  - Post-OCR korekta tekstu (NSSpellChecker) z dynamicznym słownikiem językowym
//  - Warstwa tekstowa CoreText pozycjonowana liniowo (odporność na kompresję i Ghostscript)
//

import Foundation
import PDFKit
import Vision
import CoreImage
import CoreText
import CoreGraphics
import ImageIO
import AppKit
import NaturalLanguage

// MARK: - Globalny kontekst CoreImage (Optymalizacja pamięci)
let sharedCIContext = CIContext(options: [.cacheIntermediates: false])

// MARK: - Zmienne globalne sterujące klasteryzacją
let paragraphGapFactor: CGFloat = 0.40

// MARK: - Parametry dostrojeniowe
struct OCRTuning {
    static let primaryMinTextHeight: Float = 0.005
    static let secondaryMinConfidence: Float = 0.35
    static let mergeIoUThreshold: CGFloat = 0.10
    static let bandCount = 3
    static let footnoteYThreshold: CGFloat = 0.22
    static let marginYThreshold: CGFloat = 0.05

    static var ocrUpscaleFactor: CGFloat = 1.0
    static var enableDespeckle: Bool = false

    static var diacriticSharpenRadius: Double = 0.6
    static var diacriticSharpenIntensity: Double = 1.2

    static let mergedWordMinLength = 6
    static let mergedWordMaxTotalLength = 24
    static let mergedWordMaxSegmentLength = 12
    static let minAvgSegmentLength: Double = 2.5
}

struct AutocropConfig {
    var enabled: Bool = false
    var paddingFraction: CGFloat = 0.01
    var minMarginFraction: CGFloat = 0.02
    var brightnessThreshold: CGFloat = 0.85 // Bezpieczniejszy próg dla skanów książek
}

struct Config {
    var inputPath: String
    var outputPath: String
    var dpi: CGFloat = 300
    var languages: [String] = ["pl-PL"]
    var sidecarPath: String?
    var customWords: [String] = []
    var fixDiacritics: Bool = false
    var splitMergedWords: Bool = false
    var useContextualNLP: Bool = false
    var autocrop = AutocropConfig()
    var noSplit: Bool = false
    var color: Bool = false // Nowe pole konfiguracyjne
    var lossless: Bool = false // Dodano pole bezstratnej kompresji

}

enum OCRPipelineError: Error, CustomStringConvertible {
    case cannotOpenDocument
    case contextCreationFailed
    case imageCreationFailed
    case visionFailed(Error)

    var description: String {
        switch self {
        case .cannotOpenDocument: return "Nie udało się otworzyć dokumentu PDF"
        case .contextCreationFailed: return "Nie udało się utworzyć kontekstu CG"
        case .imageCreationFailed: return "Nie udało się utworzyć obrazu z kontekstu"
        case .visionFailed(let e): return "Błąd Vision: \(e.localizedDescription)"
        }
    }
}

// MARK: - Model danych i precyzyjne mapowanie geometryczne
struct PageGeometry {
    let width: CGFloat
    let height: CGFloat
    let scale: CGFloat
    
    let drawX: CGFloat
    let drawY: CGFloat
    let drawW: CGFloat
    let drawH: CGFloat

    // Inicjalizator dla standardowego zachowania
    init(width: CGFloat, height: CGFloat, scale: CGFloat) {
        self.width = width
        self.height = height
        self.scale = scale
        self.drawX = 0
        self.drawY = 0
        self.drawW = width
        self.drawH = height
    }

    // Inicjalizator dla wycentrowanego dopasowania do płótna (Letterbox)
    init(canvasW: CGFloat, canvasH: CGFloat, imagePixelW: Int, imagePixelH: Int, scale: CGFloat) {
        self.width = canvasW
        self.height = canvasH
        self.scale = scale
        
        // Wymiary obrazka w punktach w skali renderingu
        let imgW_pts = CGFloat(imagePixelW) / scale
        let imgH_pts = CGFloat(imagePixelH) / scale
        
        // Dopasowujemy obraz do płótna z zachowaniem proporcji (Aspect Fit)
        let scaleW = canvasW / imgW_pts
        let scaleH = canvasH / imgH_pts
        let fitScale = min(scaleW, scaleH)
        
        self.drawW = imgW_pts * fitScale
        self.drawH = imgH_pts * fitScale
        
        // Wyśrodkowanie na wirtualnym płótnie
        self.drawX = (canvasW - self.drawW) / 2.0
        self.drawY = (canvasH - self.drawH) / 2.0
    }

    func map(_ normalized: CGPoint) -> CGPoint {
        return CGPoint(
            x: drawX + normalized.x * drawW,
            y: drawY + normalized.y * drawH
        )
    }
}

struct Quad {
    var topLeft: CGPoint
    var topRight: CGPoint
    var bottomLeft: CGPoint
    var bottomRight: CGPoint

    init(topLeft: CGPoint, topRight: CGPoint, bottomLeft: CGPoint, bottomRight: CGPoint) {
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
    }

    init(from obs: VNRecognizedTextObservation) {
        topLeft = obs.topLeft
        topRight = obs.topRight
        bottomLeft = obs.bottomLeft
        bottomRight = obs.bottomRight
    }

    func mapped(using geometry: PageGeometry) -> Quad {
        return Quad(
            topLeft: geometry.map(topLeft),
            topRight: geometry.map(topRight),
            bottomLeft: geometry.map(bottomLeft),
            bottomRight: geometry.map(bottomRight)
        )
    }

    var rotationAngle: CGFloat {
        return atan2(topRight.y - topLeft.y, topRight.x - topLeft.x)
    }

    var width: CGFloat {
        return hypot(topRight.x - topLeft.x, topRight.y - topLeft.y)
    }

    var height: CGFloat {
        return hypot(bottomLeft.x - topLeft.x, bottomLeft.y - topLeft.y)
    }
}

enum LineRole {
    case body
    case footnote
    case marginal
}

struct RecognizedWord {
    let text: String
    let quad: Quad
}

struct RecognizedLine {
    let text: String
    let quad: Quad
    let confidence: Float
    let words: [RecognizedWord]
    var role: LineRole = .body
}

struct Paragraph {
    var lines: [RecognizedLine]
    var readingOrderIndex: Int
    var role: LineRole
}

struct PageResult {
    let logicalPageIndex: Int
    let originalPageIndex: Int
    let geometry: PageGeometry
    let backgroundImage: CGImage
    let paragraphs: [Paragraph]
    let originalPDFPage: CGPDFPage?
}

// MARK: - Dwutorowy Autocrop (Vision ML 3D z korekcją perspektywy + Fallback Pikselowy) z pełnym raportowaniem
func applyAutocropToCGImage(_ image: CGImage, config: AutocropConfig) -> CGImage? {
    let w = CGFloat(image.width)
    let h = CGFloat(image.height)
    let logPrefix = "   [Autocrop] Strona \(Int(w))x\(Int(h)) px"
    
    var processedImage: CGImage? = nil
    var pathUsed = ""
    var widthReduction: CGFloat = 0.0
    var heightReduction: CGFloat = 0.0
    var rejectionReason = "brak jednoznacznej detekcji krawędzi papieru z obu torów"
    
    // --- TOR 1 (Główny dla zdjęć z telefonu): Apple Vision ML + Korekcja Perspektywy 3D ---
    let docRequest = VNDetectDocumentSegmentationRequest()
    docRequest.revision = VNDetectDocumentSegmentationRequestRevision1
    let docHandler = VNImageRequestHandler(cgImage: image, options: [:])
    
    if let _ = try? docHandler.perform([docRequest]),
       let results = docRequest.results,
       let document = results.first {
        
        let pts = [document.topLeft, document.topRight, document.bottomLeft, document.bottomRight]
        let sortedByX = pts.sorted { $0.x < $1.x }
        
        let leftPts = sortedByX.prefix(2).sorted { $0.y < $1.y }
        let rightPts = sortedByX.suffix(2).sorted { $0.y < $1.y }
        
        let geoBottomLeft  = leftPts[0]
        let geoTopLeft     = leftPts[1]
        let geoBottomRight = rightPts[0]
        let geoTopRight    = rightPts[1]
        
        let topLeft     = CIVector(x: geoTopLeft.x * w,     y: geoTopLeft.y * h)
        let topRight    = CIVector(x: geoTopRight.x * w,    y: geoTopRight.y * h)
        let bottomLeft  = CIVector(x: geoBottomLeft.x * w,  y: geoBottomLeft.y * h)
        let bottomRight = CIVector(x: geoBottomRight.x * w, y: geoBottomRight.y * h)
        
        let ciImage = CIImage(cgImage: image)
        
        if let filter = CIFilter(name: "CIPerspectiveCorrection") {
            filter.setValue(ciImage, forKey: kCIInputImageKey)
            filter.setValue(topLeft,     forKey: "inputTopLeft")
            filter.setValue(topRight,    forKey: "inputTopRight")
            filter.setValue(bottomLeft,  forKey: "inputBottomLeft")
            filter.setValue(bottomRight, forKey: "inputBottomRight")
            
            if let correctedImage = filter.outputImage {
                let extent = correctedImage.extent
                widthReduction = 1.0 - (extent.width / w)
                heightReduction = 1.0 - (extent.height / h)
                
                let maxAllowedWidthReduction: CGFloat = 0.50
                let maxAllowedHeightReduction: CGFloat = 0.45 // Zwiększono do 45% (ochrona przed palcami/cieniami)
                let originalRatio = w / h
                let croppedRatio = extent.width / extent.height
                let ratioDeviation = abs(originalRatio - croppedRatio) / originalRatio
                
                if widthReduction < config.minMarginFraction && heightReduction < config.minMarginFraction {
                    rejectionReason = "ML: cięcie zbyt małe (W=-\(Int(widthReduction*100))%, H=-\(Int(heightReduction*100))%)"
                } else if widthReduction > maxAllowedWidthReduction || heightReduction > maxAllowedHeightReduction {
                    rejectionReason = "ML: zbyt agresywne cięcie (W=-\(Int(widthReduction*100))%, H=-\(Int(heightReduction*100))%)"
                } else if ratioDeviation > 0.40 { // Zwiększono tolerancję zniekształcenia proporcji do 40%
                    rejectionReason = "ML: zniekształcenie proporcji bocznych o \(Int(ratioDeviation*100))%"
                } else {
                    processedImage = sharedCIContext.createCGImage(correctedImage, from: extent)
                    pathUsed = "Vision ML (3D)"
                }
            }
        }
    }
    
    // --- TOR 2 (Fallback dla skanerów płaskich): Analiza Gęstości Pikseli (2D) ---
    if processedImage == nil {
        let targetW = 200
        let targetH = 200
        let grayColorSpace = CGColorSpaceCreateDeviceGray()
        
        if let context = CGContext(
            data: nil, width: targetW, height: targetH,
            bitsPerComponent: 8, bytesPerRow: targetW,
            space: grayColorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) {
            context.draw(image, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
            
            if let data = context.data {
                let pixels = data.bindMemory(to: UInt8.self, capacity: targetW * targetH)
                let pixelThreshold = UInt8(config.brightnessThreshold * 255)
                
                // Bezpieczne, wysokie progi gęstości bieli (75% boku) - ignorują szum skanera i błędy liter
                let requiredDensityW = Int(Double(targetH) * 0.75)
                let requiredDensityH = Int(Double(targetW) * 0.75)
                
                var columnIsWhite = [Bool](repeating: false, count: targetW)
                for x in 0..<targetW {
                    var brightCount = 0
                    for y in 0..<targetH {
                        if pixels[y * targetW + x] > pixelThreshold { brightCount += 1 }
                    }
                    columnIsWhite[x] = (brightCount > requiredDensityW)
                }
                
                var rowIsWhite = [Bool](repeating: false, count: targetH)
                for y in 0..<targetH {
                    var brightCount = 0
                    for x in 0..<targetW {
                        if pixels[y * targetW + x] > pixelThreshold { brightCount += 1 }
                    }
                    rowIsWhite[y] = (brightCount > requiredDensityH)
                }
                
                // Detekcja lewej krawędzi (minX) - szukamy 3 kolejnych jasnych kolumn
                var minX = 0
                for x in 0..<(targetW - 3) {
                    if columnIsWhite[x] && columnIsWhite[x+1] && columnIsWhite[x+2] {
                        minX = x
                        break
                    }
                }
                
                // Detekcja prawej krawędzi (maxX)
                var maxX = targetW - 1
                for x in stride(from: targetW - 1, through: 2, by: -1) {
                    if columnIsWhite[x] && columnIsWhite[x-1] && columnIsWhite[x-2] {
                        maxX = x
                        break
                    }
                }
                
                // Detekcja dolnej krawędzi (minY) - szukamy 3 kolejnych jasnych rzędów
                var minY = 0
                for y in 0..<(targetH - 3) {
                    if rowIsWhite[y] && rowIsWhite[y+1] && rowIsWhite[y+2] {
                        minY = y
                        break
                    }
                }
                
                // Detekcja górnej krawędzi (maxY)
                var maxY = targetH - 1
                for y in stride(from: targetH - 1, through: 2, by: -1) {
                    if rowIsWhite[y] && rowIsWhite[y-1] && rowIsWhite[y-2] {
                        maxY = y
                        break
                    }
                }
                
                if maxX > minX && maxY > minY && (maxX - minX) > 40 && (maxY - minY) > 40 {
                    let normMinX = CGFloat(minX) / CGFloat(targetW)
                    let normMaxX = CGFloat(maxX) / CGFloat(targetW)
                    let normMinY = CGFloat(minY) / CGFloat(targetH)
                    let normMaxY = CGFloat(maxY) / CGFloat(targetH)
                    
                    var cropRect = CGRect(
                        x: normMinX * w,
                        y: normMinY * h,
                        width: (normMaxX - normMinX) * w,
                        height: (normMaxY - normMinY) * h
                    )
                    
                    let padX = w * config.paddingFraction
                    let padY = h * config.paddingFraction
                    cropRect = cropRect.insetBy(dx: -padX, dy: -padY)
                    cropRect = cropRect.intersection(CGRect(x: 0, y: 0, width: w, height: h))
                    
                    widthReduction = 1.0 - (cropRect.width / w)
                    heightReduction = 1.0 - (cropRect.height / h)
                    
                    let maxAllowedWidthReduction: CGFloat = 0.50
                    let maxAllowedHeightReduction: CGFloat = 0.45 // Zwiększono do 45%
                    let originalRatio = w / h
                    let croppedRatio = cropRect.width / cropRect.height
                    let ratioDeviation = abs(originalRatio - croppedRatio) / originalRatio
                    
                    if widthReduction < config.minMarginFraction && heightReduction < config.minMarginFraction {
                        rejectionReason = "Piksele: cięcie zbyt małe (W=-\(Int(widthReduction*100))%, H=-\(Int(heightReduction*100))%)"
                    } else if widthReduction > maxAllowedWidthReduction || heightReduction > maxAllowedHeightReduction {
                        rejectionReason = "Piksele: zbyt agresywne cięcie (W=-\(Int(widthReduction*100))%, H=-\(Int(heightReduction*100))%)"
                    } else if ratioDeviation > 0.40 { // Zwiększono do 40%
                        rejectionReason = "Piksele: zniekształcenie proporcji bocznych o \(Int(ratioDeviation*100))%"
                    } else {
                        let cgCropRect = CGRect(
                            x: cropRect.origin.x,
                            y: h - (cropRect.origin.y + cropRect.height),
                            width: cropRect.width,
                            height: cropRect.height
                        )
                        processedImage = image.cropping(to: cgCropRect)
                        pathUsed = "Piksele (2D)"
                    }
                }
            }
        }
    }
    
    // Raportowanie błędów w przypadku odrzucenia kadrowania przez bezpieczniki
    guard let resultImage = processedImage else {
        print("   ⚠️\(logPrefix): Pominięto (\(rejectionReason))")
        return nil
    }
    
    let wRedPct = Int(widthReduction * 100)
    let hRedPct = Int(heightReduction * 100)
    print("   🔍 [Autocrop] Użyto toru: \(pathUsed), wycięto: W=-\(wRedPct)%, H=-\(hRedPct)%")
    
    return resultImage
}
// MARK: - [ML] Wykrywanie rynny między stronami rozkładówki
func detectGutter(for image: CGImage, isVertical: Bool) -> Int {
    let W = image.width
    let H = image.height

    let maxW = 600
    let scale = min(1.0, Double(maxW) / Double(isVertical ? W : H))
    let sw = max(1, Int(Double(W) * scale))
    let sh = max(1, Int(Double(H) * scale))

    let colorSpace = CGColorSpaceCreateDeviceGray()
    guard let ctx = CGContext(
        data: nil, width: sw, height: sh,
        bitsPerComponent: 8, bytesPerRow: sw,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else { return isVertical ? W / 2 : H / 2 }

    ctx.draw(image, in: CGRect(x: 0, y: 0, width: sw, height: sh))
    guard let data = ctx.data else { return W / 2 }
    let pixels = data.bindMemory(to: UInt8.self, capacity: sw * sh)

    if isVertical {
        var profile = [Int](repeating: 0, count: sw)
        for y in 0..<sh {
            for x in 0..<sw {
                profile[x] += 255 - Int(pixels[y * sw + x])
            }
        }

        let win = max(2, sw / 80)
        var smoothed = profile
        for x in 0..<sw {
            var s = 0, n = 0
            for dx in -win...win {
                let nx = x + dx
                guard nx >= 0, nx < sw else { continue }
                s += profile[nx]; n += 1
            }
            smoothed[x] = n > 0 ? s / n : profile[x]
        }

        // Zawężenie obszaru wyszukiwania pionowej rynny do przedziału 45% - 55% szerokości
        let lo = Int(Double(sw) * 0.45)
        let hi = Int(Double(sw) * 0.55)
        var minVal = Int.max
        var minX  = sw / 2
        for x in lo...hi where smoothed[x] < minVal {
            minVal = smoothed[x]
            minX   = x
        }
        return Int((Double(minX) / scale).rounded())
    } else {
        var profile = [Int](repeating: 0, count: sh)
        for y in 0..<sh {
            for x in 0..<sw {
                profile[y] += 255 - Int(pixels[y * sw + x])
            }
        }

        let win = max(2, sh / 80)
        var smoothed = profile
        for y in 0..<sh {
            var s = 0, n = 0
            for dy in -win...win {
                let ny = y + dy
                guard ny >= 0, ny < sh else { continue }
                s += profile[ny]; n += 1
            }
            smoothed[y] = n > 0 ? s / n : profile[y]
        }

        // Zawężenie obszaru wyszukiwania poziomej rynny do przedziału 45% - 55% wysokości
        let lo = Int(Double(sh) * 0.45)
        let hi = Int(Double(sh) * 0.55)
        var minVal = Int.max
        var minY  = sh / 2
        for y in lo...hi where smoothed[y] < minVal {
            minVal = smoothed[y]
            minY   = y
        }
        return Int((Double(minY) / scale).rounded())
    }
}

// MARK: - Obrót tła strony bez filtrów
func rotateOnly(_ image: CGImage, by angle: CGFloat) -> CGImage {
    guard abs(angle) > 0.002 else { return image }
    var ci = CIImage(cgImage: image)
    ci = ci.transformed(by: CGAffineTransform(rotationAngle: -angle))
    return sharedCIContext.createCGImage(ci, from: ci.extent) ?? image
}

// MARK: - Programistyczna detekcja rzeczywistej orientacji tekstu
func detectTrueTextOrientation(for image: CGImage, languages: [String]) async -> (image: CGImage, wasRotated: Bool, angleApplied: CGFloat) {
    guard let quickLines = try? await recognizeText(
        in: image,
        languages: languages,
        minimumTextHeight: 0.018, // Obniżono z 0.035 do 0.018, aby silnik widział nagłówki i większy tekst książki
        customWords: []
    ), quickLines.count >= 3 else { // Bezpiecznik: minimum 3 linie (chroni puste strony i okładki)
        return (image, false, 0)
    }
    
    let angles = quickLines.map { $0.quad.rotationAngle }.sorted()
    let medianAngle = angles[angles.count / 2]
    
    let absAngle = abs(medianAngle)
    if absAngle > 0.78 && absAngle < 2.35 {
        let rotationAngle: CGFloat = medianAngle > 0 ? (CGFloat.pi / 2) : (-CGFloat.pi / 2)
        let rotated = rotateOnly(image, by: rotationAngle)
        return (rotated, true, rotationAngle)
    } else if absAngle >= 2.35 {
        let rotationAngle = CGFloat.pi
        let rotated = rotateOnly(image, by: rotationAngle)
        return (rotated, true, rotationAngle)
    }
    
    return (image, false, 0)
}
// MARK: - Natywne, wątkowo bezpieczne renderowanie strony do CGImage (Skala szarości lub opcjonalny sRGB)
func renderPDFPageToCGImage(_ page: PDFPage, scale: CGFloat, color: Bool = false) -> CGImage? {
    guard let cgPage = page.pageRef else { return nil }
    
    let bounds = page.bounds(for: .cropBox)
    let rotation = page.rotation
    let isRotated = (rotation == 90 || rotation == 270)
    let displayWidth  = Int((isRotated ? bounds.height : bounds.width) * scale)
    let displayHeight = Int((isRotated ? bounds.width : bounds.height) * scale)
    
    let cgColorSpace: CGColorSpace
    let cgBitmapInfo: UInt32
    let bytesPerRow: Int
    
    if color {
        // Konfiguracja kolorowa (32-bit sRGB, 4 bajty na piksel)
        cgColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        cgBitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        bytesPerRow = displayWidth * 4
    } else {
        // Domyślna konfiguracja lekka (8-bit DeviceGray, 1 bajt na piksel - 75% oszczędności RAM)
        cgColorSpace = CGColorSpaceCreateDeviceGray()
        cgBitmapInfo = CGImageAlphaInfo.none.rawValue
        bytesPerRow = displayWidth
    }
    
    guard let context = CGContext(
        data: nil,
        width: displayWidth,
        height: displayHeight,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: cgColorSpace,
        bitmapInfo: cgBitmapInfo
    ) else { return nil }
    
    if color {
        context.setFillColor(NSColor.white.cgColor)
    } else {
        context.setFillColor(gray: 1.0, alpha: 1.0)
    }
    context.fill(CGRect(x: 0, y: 0, width: displayWidth, height: displayHeight))
    
    context.saveGState()
    context.scaleBy(x: scale, y: scale)
    
    // Zunifikowana i poprawna transformacja obrotu Y-up
    switch rotation {
    case 90:
        context.translateBy(x: bounds.height, y: 0)
        context.rotate(by: .pi / 2)
    case 180:
        context.translateBy(x: bounds.width, y: bounds.height)
        context.rotate(by: .pi)
    case 270:
        context.translateBy(x: 0, y: bounds.width)
        context.rotate(by: -.pi / 2)
    default:
        break
    }
    
    context.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
    context.drawPDFPage(cgPage)
    context.restoreGState()
    
    return context.makeImage()
}
// MARK: - Renderowanie strony i automatyczny Split Rozkładówek (Zoptymalizowany pod kątem ochrony tabel poziomych)
func renderPagesSplitting(_ page: PDFPage, scale: CGFloat, originalIndex: Int, autocrop: AutocropConfig, languages: [String], color: Bool = false) async throws -> [(image: CGImage, geometry: PageGeometry, wasRotated: Bool)] {
    
    let bounds = page.bounds(for: .cropBox)
    let originalIsLandscape = bounds.width > bounds.height * 1.05
    
    // Obliczamy współczynnik dopasowania (fitScale), aby po podziale uzyskać standardowy format pionowy A4 (595 x 842 pkt)
    let fitScale: CGFloat
    if originalIsLandscape {
        let halfWidth = bounds.width / 2.0
        let scaleW = 595.0 / halfWidth
        let scaleH = 842.0 / bounds.height
        fitScale = min(scaleW, scaleH)
    } else {
        let scaleW = 595.0 / bounds.width
        let scaleH = 842.0 / bounds.height
        fitScale = min(scaleW, scaleH)
    }
    
    let effectiveScale = scale * fitScale
    
    guard let cgImage = renderPDFPageToCGImage(page, scale: effectiveScale, color: color) else {
        throw OCRPipelineError.imageCreationFailed
    }
    
    var uprightImage = cgImage
    var wasRotated = false
    
    // DETERMINISTYCZNY BEZPIECZNIK GEOMETRYCZNY:
    if originalIsLandscape && cgImage.width < cgImage.height {
        uprightImage = rotateOnly(cgImage, by: .pi / 2)
        wasRotated = true
    } else {
        let (img, rotated, _) = await detectTrueTextOrientation(for: cgImage, languages: languages)
        uprightImage = img
        wasRotated = rotated
    }
    
    let isLandscape = originalIsLandscape && (CGFloat(uprightImage.width) > CGFloat(uprightImage.height) * 1.05)
    
    let textSkew = await detectTextSkewAngle(for: uprightImage, languages: languages)
    var processedSpread = rotateOnly(uprightImage, by: textSkew)
    
    if autocrop.enabled {
        if let cropped = applyAutocropToCGImage(processedSpread, config: autocrop) {
            processedSpread = cropped
        }
    }
    
    let pixelWidth = processedSpread.width
    let pixelHeight = processedSpread.height
    
    if isLandscape {
        // Inteligentne lokalizowanie fizycznego grzbietu książki
        let cutX = detectGutter(for: processedSpread, isVertical: true)
        
        let leftRect  = CGRect(x: 0,    y: 0, width: cutX, height: pixelHeight)
        let rightRect = CGRect(x: cutX, y: 0, width: pixelWidth - cutX, height: pixelHeight)
        
        guard let leftImg = processedSpread.cropping(to: leftRect),
              let rightImg = processedSpread.cropping(to: rightRect) else {
            throw OCRPipelineError.imageCreationFailed
        }
        
        // Tworzymy idealnie równe, wycentrowane na wirtualnym płótnie A4 (595 x 842) geometrie stron
        let leftGeom = PageGeometry(canvasW: 595.0, canvasH: 842.0, imagePixelW: leftImg.width, imagePixelH: leftImg.height, scale: effectiveScale)
        let rightGeom = PageGeometry(canvasW: 595.0, canvasH: 842.0, imagePixelW: rightImg.width, imagePixelH: rightImg.height, scale: effectiveScale)
        
        return [(leftImg, leftGeom, wasRotated), (rightImg, rightGeom, wasRotated)]
    } else {
        let geom = PageGeometry(canvasW: 595.0, canvasH: 842.0, imagePixelW: pixelWidth, imagePixelH: pixelHeight, scale: effectiveScale)
        return [(processedSpread, geom, wasRotated)]
    }
}

// MARK: - Renderowanie strony bez podziału (--nosplit)
func renderPageNoSplit(_ page: PDFPage, scale: CGFloat, color: Bool = false) throws -> (image: CGImage, geometry: PageGeometry, pdfPage: CGPDFPage?) {
    return try autoreleasepool {
        let bounds = page.bounds(for: .cropBox)
        let rotation = page.rotation
        let isRotated = (rotation == 90 || rotation == 270)
        let originalWidth  = isRotated ? bounds.height : bounds.width
        let originalHeight = isRotated ? bounds.width  : bounds.height

        let isLandscape = originalWidth > originalHeight * 1.05
        
        // Obliczamy skalę dopasowania do formatu A4
        let targetWidth = isLandscape ? 842.0 : 595.0
        let targetHeight = isLandscape ? 595.0 : 842.0
        
        let scaleW = targetWidth / originalWidth
        let scaleH = targetHeight / originalHeight
        let fitScale = min(scaleW, scaleH)
        
        let effectiveScale = scale * fitScale

        guard let cgImage = renderPDFPageToCGImage(page, scale: effectiveScale, color: color) else {
            throw OCRPipelineError.imageCreationFailed
        }
        
        // Tworzymy wycentrowaną geometrię strony na płótnie A4
        let geom = PageGeometry(canvasW: targetWidth, canvasH: targetHeight, imagePixelW: cgImage.width, imagePixelH: cgImage.height, scale: effectiveScale)
        
        return (cgImage, geom, nil)
    }
}
// MARK: - [ML] Pomocnicze : Horizon & Super-Resolution
func detectHorizonAngle(for image: CGImage) -> CGFloat {
    let request = VNDetectHorizonRequest()
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
        if let result = request.results?.first as? VNHorizonObservation {
            return result.angle
        }
    } catch {}
    return 0.0
}

func applySuperResolutionML(image: CGImage) -> CGImage {
    if #available(macOS 15.0, *) {
        if let requestClass = NSClassFromString("VNGenerateImageSuperResolutionRequest") as? NSObject.Type {
            let request = requestClass.init() as! VNRequest
            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
                if let results = request.value(forKey: "results") as? [NSObject],
                   let firstResult = results.first,
                   let pb = firstResult.value(forKey: "pixelBuffer") {
                    let pbType = CFGetTypeID(pb as CFTypeRef)
                    if pbType == CVPixelBufferGetTypeID() {
                        let ciImage = CIImage(cvPixelBuffer: pb as! CVPixelBuffer)
                        if let cgOut = sharedCIContext.createCGImage(ciImage, from: ciImage.extent) {
                            return cgOut
                        }
                    }
                }
            } catch {}
        }
    }
    return image
}

// MARK: - [ML] Klasyfikacja zawartości strony (fotografia/ilustracja vs. tekst)
struct PageContentClassification {
    let topLabel: String
    let topConfidence: Float
    let allLabels: [(label: String, confidence: Float)]
}

func classifyPageVisualContent(_ image: CGImage) -> PageContentClassification? {
    let request = VNClassifyImageRequest()
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
        guard let results = request.results, !results.isEmpty else { return nil }
        let sorted = results.sorted { $0.confidence > $1.confidence }
        let top = sorted[0]
        let all = sorted.prefix(10).map { (label: $0.identifier, confidence: $0.confidence) }
        return PageContentClassification(topLabel: top.identifier, topConfidence: top.confidence, allLabels: Array(all))
    } catch {
        return nil
    }
}

func isLikelyPhotographicPage(
    _ classification: PageContentClassification,
    photoLikeKeywords: Set<String> = ["photo", "illustration", "drawing", "painting", "portrait", "art", "poster"],
    minConfidence: Float = 0.35
) -> Bool {
    guard classification.topConfidence >= minConfidence else { return false }
    let label = classification.topLabel.lowercased()
    return photoLikeKeywords.contains { label.contains($0) }
}

// MARK: - [ML] Ocena jakości/charakteru skanu (adaptacyjny upscale)
struct PageAestheticsResult {
    let overallScore: Float
    let isUtility: Bool
}

@available(macOS 15.0, *)
func evaluateScanAesthetics(_ image: CGImage) -> PageAestheticsResult? {
    let request = VNCalculateImageAestheticsScoresRequest()
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
        guard let result = request.results?.first as? VNImageAestheticsScoresObservation else { return nil }
        return PageAestheticsResult(overallScore: result.overallScore, isUtility: result.isUtility)
    } catch {
        return nil
    }
}

@available(macOS 15.0, *)
func adaptiveUpscaleFactor(for image: CGImage, baseFactor: CGFloat, weakScanThreshold: Float = -0.2) -> CGFloat {
    guard let aesthetics = evaluateScanAesthetics(image) else { return baseFactor }
    if aesthetics.overallScore < weakScanThreshold {
        return max(baseFactor, 1.4)
    }
    return baseFactor
}

// MARK: - Preprocessing obrazu pod OCR
func preprocessForOCR(_ image: CGImage, skewAngle: CGFloat = 0) -> CGImage {
    return autoreleasepool {
        var working = CIImage(cgImage: image)

        if abs(skewAngle) > 0.002 {
            working = working.transformed(by: CGAffineTransform(rotationAngle: -skewAngle))
        }

        var processedImage: CGImage
        if let currentCG = sharedCIContext.createCGImage(working, from: working.extent) {
            processedImage = currentCG
        } else {
            processedImage = image
        }

        var effectiveUpscale = OCRTuning.ocrUpscaleFactor
        if #available(macOS 15.0, *) {
            effectiveUpscale = adaptiveUpscaleFactor(for: image, baseFactor: OCRTuning.ocrUpscaleFactor)
        }

        if effectiveUpscale > 1.001 {
            let superResImage = applySuperResolutionML(image: processedImage)
            if superResImage !== processedImage {
                processedImage = superResImage
                working = CIImage(cgImage: processedImage)
            } else {
                if let scaleFilter = CIFilter(name: "CILanczosScaleTransform") {
                    scaleFilter.setValue(working, forKey: kCIInputImageKey)
                    scaleFilter.setValue(effectiveUpscale, forKey: kCIInputScaleKey)
                    scaleFilter.setValue(1.0, forKey: kCIInputAspectRatioKey)
                    if let out = scaleFilter.outputImage {
                        working = out
                    }
                }
            }
        }

        if OCRTuning.enableDespeckle, let median = CIFilter(name: "CIMedianFilter") {
            median.setValue(working, forKey: kCIInputImageKey)
            if let out = median.outputImage {
                working = out
            }
        }

        if let hs = CIFilter(name: "CIHighlightShadowAdjust") {
            hs.setValue(working, forKey: kCIInputImageKey)
            hs.setValue(0.75, forKey: "inputRadius")
            hs.setValue(0.6, forKey: "inputShadowAmount")
            hs.setValue(1.0, forKey: "inputHighlightAmount")
            if let out = hs.outputImage {
                working = out
            }
        }

        if let cc = CIFilter(name: "CIColorControls") {
            cc.setValue(working, forKey: kCIInputImageKey)
            cc.setValue(0.0, forKey: kCIInputSaturationKey)
            cc.setValue(1.15, forKey: kCIInputContrastKey)
            cc.setValue(0.0, forKey: kCIInputBrightnessKey)
            if let out = cc.outputImage {
                working = out
            }
        }
        if let dilate = CIFilter(name: "CIMorphologyRectangleMaximum"),
           let erode = CIFilter(name: "CIMorphologyRectangleMinimum") {
            let morphPx = 3.0 as NSNumber

            dilate.setValue(working, forKey: kCIInputImageKey)
            dilate.setValue(morphPx, forKey: "inputWidth")
            dilate.setValue(morphPx, forKey: "inputHeight")

            if let dilated = dilate.outputImage {
                erode.setValue(dilated, forKey: kCIInputImageKey)
                erode.setValue(morphPx, forKey: "inputWidth")
                erode.setValue(morphPx, forKey: "inputHeight")
                if let closed = erode.outputImage {
                    working = closed
                }
            }
        }

        let dpi: CGFloat = 300
        let dpiScale = Double(dpi / 300.0)
        let adaptiveRadius = 0.6 * dpiScale
        let adaptiveIntensity = 1.2 / max(1.0, dpiScale * 0.6)

        if let unsharp = CIFilter(name: "CIUnsharpMask") {
            unsharp.setValue(working, forKey: kCIInputImageKey)
            unsharp.setValue(adaptiveRadius, forKey: kCIInputRadiusKey)
            unsharp.setValue(adaptiveIntensity, forKey: kCIInputIntensityKey)
            if let out = unsharp.outputImage { working = out }
        }

        guard let cgOut = sharedCIContext.createCGImage(working, from: working.extent) else { return processedImage }
        return cgOut
    }
}

// MARK: - Ekstrakcja boxów per-słowo
func aabb(_ q: Quad) -> CGRect {
    let xs = [q.topLeft.x, q.topRight.x, q.bottomLeft.x, q.bottomRight.x]
    let ys = [q.topLeft.y, q.topRight.y, q.bottomLeft.y, q.bottomRight.y]
    let minX = xs.min() ?? 0, maxX = xs.max() ?? 0
    let minY = ys.min() ?? 0, maxY = ys.max() ?? 0
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
}

func quadIoU(_ a: Quad, _ b: Quad) -> CGFloat {
    let ra = aabb(a)
    let rb = aabb(b)
    let intersection = ra.intersection(rb)
    if intersection.isNull || intersection.width <= 0 || intersection.height <= 0 { return 0 }
    let interArea = intersection.width * intersection.height
    let unionArea = ra.width * ra.height + rb.width * rb.height - interArea
    return unionArea > 0 ? interArea / unionArea : 0
}

func mergeOCRPasses(primary: [RecognizedLine], secondary: [RecognizedLine]) -> [RecognizedLine] {
    guard !secondary.isEmpty else { return primary }

    let secondaryFiltered = secondary.filter { $0.confidence >= OCRTuning.secondaryMinConfidence }

    var result = primary
    for sLine in secondaryFiltered {
        let overlapsExisting = primary.contains { pLine in
            quadIoU(sLine.quad, pLine.quad) > OCRTuning.mergeIoUThreshold
        }
        if !overlapsExisting {
            result.append(sLine)
        }
    }
    return result
}

// MARK: - Transformacje kątowe i rynna
func calculateSkew(from lines: [RecognizedLine]) -> CGFloat {
    guard !lines.isEmpty else { return 0 }
    let angles = lines.map { $0.quad.rotationAngle }.sorted()
    return angles[angles.count / 2]
}

func deskewedX(point: CGPoint, skew: CGFloat) -> CGFloat {
    let dx = point.x - 0.5
    let dy = point.y - 0.5
    return dx * cos(-skew) - dy * sin(-skew) + 0.5
}

func deskewedXRange(quad: Quad, skew: CGFloat) -> (minX: CGFloat, maxX: CGFloat) {
    let pts = [quad.topLeft, quad.topRight, quad.bottomLeft, quad.bottomRight]
    let xs = pts.map { deskewedX(point: $0, skew: skew) }
    return (xs.min() ?? 0, xs.max() ?? 1)
}

func deskewedXCenter(quad: Quad, skew: CGFloat) -> CGFloat {
    let cx = (quad.topLeft.x + quad.topRight.x + quad.bottomLeft.x + quad.bottomRight.x) / 4.0
    let cy = (quad.topLeft.y + quad.topRight.y + quad.bottomLeft.y + quad.bottomRight.y) / 4.0
    let dx = cx - 0.5
    let dy = cy - 0.5
    return dx * cos(-skew) - dy * sin(-skew) + 0.5
}

func lineCenterY(_ line: RecognizedLine) -> CGFloat {
    return (line.quad.topLeft.y + line.quad.topRight.y + line.quad.bottomLeft.y + line.quad.bottomRight.y) / 4.0
}

func detectColumnGutterStrict(from lines: [RecognizedLine], skew: CGFloat) -> CGFloat? {
    let filteredLines = lines.filter { $0.quad.width <= 0.80 }
    guard filteredLines.count >= 6 else { return nil }

    let bins = 200
    var coverage = [Int](repeating: 0, count: bins)

    for line in filteredLines {
        let (minX, maxX) = deskewedXRange(quad: line.quad, skew: skew)
        let startBin = max(0, min(bins - 1, Int(minX * CGFloat(bins))))
        let endBin = max(0, min(bins - 1, Int(maxX * CGFloat(bins))))
        for i in startBin...endBin { coverage[i] += 1 }
    }

    let lowBound = Int(0.30 * CGFloat(bins))
    let highBound = Int(0.70 * CGFloat(bins))

    var totalCoverage = 0
    for i in lowBound...highBound {
        totalCoverage += coverage[i]
    }
    let avgCoverage = Double(totalCoverage) / Double(highBound - lowBound + 1)
    let threshold = max(1, Int(avgCoverage * 0.15))

    var bestStart = -1
    var bestLen = 0
    var i = lowBound
    while i <= highBound {
        if coverage[i] <= threshold {
            var j = i
            while j <= highBound && coverage[j] <= threshold { j += 1 }
            if j - i > bestLen {
                bestLen = j - i
                bestStart = i
            }
            i = j
        } else {
            i += 1
        }
    }

    let minGapBins = max(2, Int(0.02 * CGFloat(bins)))
    guard bestStart >= 0, bestLen >= minGapBins else { return nil }

    let gapCenter = CGFloat(bestStart) + CGFloat(bestLen) / 2.0
    return gapCenter / CGFloat(bins)
}

// MARK: - Deskew pasmowy (per-band)
struct Band {
    let yMin: CGFloat
    let yMax: CGFloat
    let skew: CGFloat
}

func computeBandedSkew(lines: [RecognizedLine], bandCount: Int) -> [Band] {
    guard !lines.isEmpty else {
        return [Band(yMin: 0, yMax: 1, skew: 0)]
    }

    let globalSkew = calculateSkew(from: lines)
    var bands: [Band] = []
    let step: CGFloat = 1.0 / CGFloat(bandCount)

    for i in 0..<bandCount {
        let yMin = CGFloat(i) * step
        let yMax = (i == bandCount - 1) ? 1.0 : yMin + step
        let bandLines = lines.filter { line in
            let cy = lineCenterY(line)
            return cy >= yMin && cy <= yMax
        }
        let skew = bandLines.count >= 4 ? calculateSkew(from: bandLines) : globalSkew
        bands.append(Band(yMin: yMin, yMax: yMax, skew: skew))
    }
    return bands
}

func bandIndex(forY cy: CGFloat, bands: [Band]) -> Int {
    for (i, band) in bands.enumerated() {
        if cy >= band.yMin && cy <= band.yMax {
            return i
        }
    }
    return bands.count - 1
}

// MARK: - Izolacja kolumn z rynną liczoną per pasmo
func assignColumns(lines: [RecognizedLine], bands: [Band]) -> [[RecognizedLine]] {
    let midSkew = bands[bands.count / 2].skew
    let globalGutter = detectColumnGutterStrict(from: lines, skew: midSkew)

    var bandGutters: [CGFloat?] = []
    for band in bands {
        let bandLines = lines.filter { line in
            let cy = lineCenterY(line)
            return cy >= band.yMin && cy <= band.yMax
        }
        let gutter = detectColumnGutterStrict(from: bandLines, skew: band.skew) ?? globalGutter
        bandGutters.append(gutter)
    }

    var left: [RecognizedLine] = []
    var right: [RecognizedLine] = []

    for line in lines {
        let cy = lineCenterY(line)
        let idx = bandIndex(forY: cy, bands: bands)
        let skew = bands[idx].skew

        if let gutter = bandGutters[idx] {
            let cx = deskewedXCenter(quad: line.quad, skew: skew)
            if cx < gutter {
                left.append(line)
            } else {
                right.append(line)
            }
        } else {
            left.append(line)
        }
    }
    return [left, right]
}

// MARK: - Klasyfikacja ról linii
func classifyLineRoles(lines: [RecognizedLine]) -> [RecognizedLine] {
    guard lines.count > 4 else { return lines }

    let heights = lines.map { $0.quad.height }.sorted()
    let medianHeight = heights[heights.count / 2]

    let centralLines = lines.filter { line in
        let cy = lineCenterY(line)
        return cy > OCRTuning.marginYThreshold && cy < (1.0 - OCRTuning.marginYThreshold)
    }
    
    let referenceLines = centralLines.isEmpty ? lines : centralLines

    let bodyTopY = referenceLines.map { lineCenterY($0) }.max() ?? 1.0
    let bodyBottomY = referenceLines.map { lineCenterY($0) }.min() ?? 0.0

    return lines.map { line -> RecognizedLine in
        var l = line
        let cy = lineCenterY(line)

        let inTopMargin = cy >= (1.0 - OCRTuning.marginYThreshold)
        let inBottomMargin = cy <= OCRTuning.marginYThreshold

        if (inTopMargin && cy > bodyTopY) || (inBottomMargin && cy < bodyBottomY) {
            l.role = .marginal
            return l
        }

        if cy < OCRTuning.footnoteYThreshold && line.quad.height < medianHeight * 0.82 {
            l.role = .footnote
            return l
        }

        l.role = .body
        return l
    }
}

// MARK: - Stabilna klasteryzacja linii w akapity wewnątrz kolumny
func horizontallyOverlaps(_ para: [RecognizedLine], _ line: RecognizedLine) -> Bool {
    let pMinX = para.map { min($0.quad.topLeft.x, $0.quad.bottomLeft.x) }.min() ?? 0
    let pMaxX = para.map { max($0.quad.topRight.x, $0.quad.bottomRight.x) }.max() ?? 1
    let lMinX = min(line.quad.topLeft.x, line.quad.bottomLeft.x)
    let lMaxX = max(line.quad.topRight.x, line.quad.bottomRight.x)
    return max(pMinX, lMinX) <= min(pMaxX, lMaxX)
}

func clusterIntoParagraphs(lines: [RecognizedLine]) -> [[RecognizedLine]] {
    guard !lines.isEmpty else { return [] }

    let sorted = lines.sorted { a, b in
        let aTop = max(a.quad.topLeft.y, a.quad.topRight.y)
        let aBottom = min(a.quad.bottomLeft.y, a.quad.bottomRight.y)
        let bTop = max(b.quad.topLeft.y, b.quad.topRight.y)
        let bBottom = min(b.quad.bottomLeft.y, b.quad.bottomRight.y)

        let overlap = min(aTop, bTop) - max(aBottom, bBottom)
        let minHeight = min(aTop - aBottom, bTop - bBottom)

        if overlap > 0.3 * minHeight {
            return a.quad.topLeft.x < b.quad.topLeft.x
        }
        return aTop > bTop
    }

    let heights = sorted.map { $0.quad.height }.sorted()
    let medianHeight = heights[heights.count / 2]

    var paragraphs: [[RecognizedLine]] = []
    var current: [RecognizedLine] = [sorted[0]]

    for i in 1..<sorted.count {
        let prev = sorted[i - 1]
        let line = sorted[i]

        let prevBottom = min(prev.quad.bottomLeft.y, prev.quad.bottomRight.y)
        let currTop = max(line.quad.topLeft.y, line.quad.topRight.y)
        let gap = prevBottom - currTop

        let isVerticalGapTooLarge = gap > medianHeight * paragraphGapFactor
        let isSideBySide = gap < -medianHeight * 0.5
        let hasHOverlap = horizontallyOverlaps(current, line)

        if isVerticalGapTooLarge || isSideBySide || !hasHOverlap {
            paragraphs.append(current)
            current = [line]
        } else {
            current.append(line)
        }
    }
    paragraphs.append(current)
    return paragraphs
}

// MARK: - Globalna funkcja rozbijania linii na słowa
func extractWords(from candidate: VNRecognizedText) -> [RecognizedWord] {
    let fullText = candidate.string
    guard fullText.contains(" ") else { return [] }

    let pieces = fullText.split(separator: " ", omittingEmptySubsequences: true)
    guard pieces.count > 1 else { return [] }

    var words: [RecognizedWord] = []
    var searchStart = fullText.startIndex

    for piece in pieces {
        let pieceStr = String(piece)
        guard let range = fullText.range(of: pieceStr, range: searchStart..<fullText.endIndex) else {
            continue
        }
        do {
            if let rect = try candidate.boundingBox(for: range) {
                let quad = Quad(
                    topLeft: rect.topLeft,
                    topRight: rect.topRight,
                    bottomLeft: rect.bottomLeft,
                    bottomRight: rect.bottomRight
                )
                words.append(RecognizedWord(text: pieceStr, quad: quad))
            }
        } catch {
        }
        searchStart = range.upperBound
    }

    return words.count == pieces.count ? words : []
}

// MARK: - Globalna funkcja Vision OCR
func recognizeText(
    in image: CGImage,
    languages: [String],
    minimumTextHeight: Float,
    customWords: [String] = []
) async throws -> [RecognizedLine] {
    try await withCheckedThrowingContinuation { continuation in
        var resumed = false

        let request = VNRecognizeTextRequest { req, error in
            guard !resumed else { return }
            resumed = true
            if let error = error {
                continuation.resume(throwing: OCRPipelineError.visionFailed(error))
                return
            }
            let observations = (req.results as? [VNRecognizedTextObservation]) ?? []
            let lines = observations.compactMap { obs -> RecognizedLine? in
                guard let candidate = obs.topCandidates(1).first else { return nil }
                let quad = Quad(from: obs)
                guard quad.width > 0.004, quad.height > 0.0008 else { return nil }
                let words = extractWords(from: candidate)
                return RecognizedLine(text: candidate.string, quad: quad,
                                      confidence: candidate.confidence, words: words)
            }
            continuation.resume(returning: lines)
        }

        request.revision = VNRecognizeTextRequestRevision3
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = languages
        request.minimumTextHeight = minimumTextHeight
        if !customWords.isEmpty { request.customWords = customWords }

        do {
            try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        } catch {
            guard !resumed else { return }
            resumed = true
            continuation.resume(throwing: OCRPipelineError.visionFailed(error))
        }
    }
}

// MARK: - Precyzyjne wykrywanie kąta pochylenia tekstu (Deskew)
func detectTextSkewAngle(
    for image: CGImage,
    languages: [String]
) async -> CGFloat {
    guard let quickLines = try? await recognizeText(
        in: image,
        languages: languages,
        minimumTextHeight: 0.012, // Obniżono do 1.2% wysokości, by widzieć zwykły tekst książki i go prostować
        customWords: []
    ), !quickLines.isEmpty else { return 0 }

    let rawSkew = calculateSkew(from: quickLines)
    
    // FILTR KĄTA: Korekta mikro-pochylenia linii skanu nigdy nie przekracza 10 stopni (0.17 rad).
    let maxAngleAllowed: CGFloat = 10.0 * .pi / 180.0
    if abs(rawSkew) > maxAngleAllowed {
        return 0
    }
    return rawSkew
}

// MARK: - Przetwarzanie OCR jednej Logicznej Strony (Architektura: Prostowanie -> Kadrowanie)
func processPageOCR(
    image: CGImage,
    geometry: PageGeometry,
    logicalPageIndex: Int,
    originalPageIndex: Int,
    languages: [String],
    customWords: [String],
    pdfPage: CGPDFPage? = nil
) async throws -> PageResult {
    let config = parseArguments()
    
    // 1. PROSTOWANIE (Deskew) — Wykonywane jako pierwsze na pełnym obrazie
    let textSkew = await detectTextSkewAngle(for: image, languages: languages)
    let rotatedBackground = rotateOnly(image, by: textSkew)
    
    var workingImage = rotatedBackground
    var workingGeometry = geometry
    var finalPDFPage = pdfPage

    // Jeśli strona miała wykryte pochylenie, wyłączamy wektorowy podkład PDF
    if abs(textSkew) > 0.005 {
        finalPDFPage = nil
    }

    // 2. KADROWANIE (Autocrop) — Wykonywane jako drugie, na idealnie spionowanym obrazie
        if config.autocrop.enabled {
            if let cropped = applyAutocropToCGImage(workingImage, config: config.autocrop) {
                workingImage = cropped
                let isLandscape = CGFloat(cropped.width) > CGFloat(cropped.height) * 1.05
                let targetW = isLandscape ? 842.0 : 595.0
                let targetH = isLandscape ? 595.0 : 842.0
                workingGeometry = PageGeometry(
                    canvasW: targetW,
                    canvasH: targetH,
                    imagePixelW: cropped.width,
                    imagePixelH: cropped.height,
                    scale: config.dpi / 72.0
                )
                finalPDFPage = nil
            }
        }

    // Preprocessing oczyszczania pod OCR na wyprostowanym i wykadrowanym obrazie
    let ocrImage = preprocessForOCR(workingImage, skewAngle: 0)

    // Pobieramy wyniki dokładnego, pojedynczego przebiegu Vision OCR.
        // Przekazujemy je bezpośrednio do klasyfikacji ról, eliminując operację merge,
        // która przy tożsamym zestawie danych wejściowych generowała ryzyko dublowania linii.
        let primaryLines = try await recognizeText(
            in: ocrImage,
            languages: languages,
            minimumTextHeight: OCRTuning.primaryMinTextHeight,
            customWords: customWords
        )

        let classified = classifyLineRoles(lines: primaryLines)

    let bodyLines = classified.filter { $0.role == .body }
    let footnoteLines = classified.filter { $0.role == .footnote }
    let marginalLines = classified.filter { $0.role == .marginal }

    let bandSource = bodyLines.isEmpty ? classified : bodyLines
    let bands = computeBandedSkew(lines: bandSource, bandCount: OCRTuning.bandCount)

    var finalParagraphs: [Paragraph] = []
        var orderIndex = 0

        let midSkew = bands[bands.count / 2].skew
        let globalGutter = detectColumnGutterStrict(from: classified, skew: midSkew)

        if let gutter = globalGutter {
            // Podział linii według wykrytej rynny (lewa i prawa strona logiczna rozkładówki)
            let leftBody = bodyLines.filter { deskewedXCenter(quad: $0.quad, skew: midSkew) < gutter }
            let rightBody = bodyLines.filter { deskewedXCenter(quad: $0.quad, skew: midSkew) >= gutter }
            
            let leftFootnotes = footnoteLines.filter { deskewedXCenter(quad: $0.quad, skew: midSkew) < gutter }
            let rightFootnotes = footnoteLines.filter { deskewedXCenter(quad: $0.quad, skew: midSkew) >= gutter }
            
            let leftMarginal = marginalLines.filter { deskewedXCenter(quad: $0.quad, skew: midSkew) < gutter }
            let rightMarginal = marginalLines.filter { deskewedXCenter(quad: $0.quad, skew: midSkew) >= gutter }
            
            // 1. Dodanie całej lewej strony logicznej (tekst główny -> przypisy -> marginalia)
            for pLines in clusterIntoParagraphs(lines: leftBody) {
                finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .body))
                orderIndex += 1
            }
            for pLines in clusterIntoParagraphs(lines: leftFootnotes) {
                finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .footnote))
                orderIndex += 1
            }
            for pLines in clusterIntoParagraphs(lines: leftMarginal) {
                finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .marginal))
                orderIndex += 1
            }
            
            // 2. Dodanie całej prawej strony logicznej (tekst główny -> przypisy -> marginalia)
            for pLines in clusterIntoParagraphs(lines: rightBody) {
                finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .body))
                orderIndex += 1
            }
            for pLines in clusterIntoParagraphs(lines: rightFootnotes) {
                finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .footnote))
                orderIndex += 1
            }
            for pLines in clusterIntoParagraphs(lines: rightMarginal) {
                finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .marginal))
                orderIndex += 1
            }
        } else {
            // Domyślna ścieżka dla braku wyraźnego podziału kolumnowego (jedna strona)
            if !bodyLines.isEmpty {
                for pLines in clusterIntoParagraphs(lines: bodyLines) {
                    finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .body))
                    orderIndex += 1
                }
            }
            if !footnoteLines.isEmpty {
                for pLines in clusterIntoParagraphs(lines: footnoteLines) {
                    finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .footnote))
                    orderIndex += 1
                }
            }
            if !marginalLines.isEmpty {
                for pLines in clusterIntoParagraphs(lines: marginalLines) {
                    finalParagraphs.append(Paragraph(lines: pLines, readingOrderIndex: orderIndex, role: .marginal))
                    orderIndex += 1
                }
            }
        }

    return PageResult(
        logicalPageIndex: logicalPageIndex,
        originalPageIndex: originalPageIndex,
        geometry: workingGeometry,
        backgroundImage: workingImage,
        paragraphs: finalParagraphs,
        originalPDFPage: finalPDFPage
    )
}

// MARK: - Rysowanie warstwy tekstowej PDF
func fontSizeForLineHeight(_ lineHeight: CGFloat, baseFont: CTFont) -> CGFloat {
    let unitFont = CTFontCreateCopyWithAttributes(baseFont, 1.0, nil, nil)
    let metrics = CTFontGetAscent(unitFont) + CTFontGetDescent(unitFont)
    return metrics > 0 ? lineHeight / metrics : lineHeight
}

func drawTextRun(text: String, quad: Quad, geometry: PageGeometry, ctx: CGContext, baseFont: CTFont, fontSize: CGFloat? = nil) {
    guard !text.isEmpty else { return }
    let mappedQuad = quad.mapped(using: geometry)
    guard mappedQuad.width > 0, mappedQuad.height > 0 else { return }

    // Używamy ujednoliconego rozmiaru akapitu lub wyliczamy go jako fallback
    let size = fontSize ?? fontSizeForLineHeight(mappedQuad.height, baseFont: baseFont)
    
    // Standardowa czcionka Helvetica gwarantująca idealne mapowanie w MS Word
    let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)

    let attrString = NSAttributedString(string: text, attributes: [
        .font: font,
        .foregroundColor: NSColor.black.withAlphaComponent(0.0)
    ])
    let ctLine = CTLineCreateWithAttributedString(attrString)

    let naturalWidth = CGFloat(CTLineGetTypographicBounds(ctLine, nil, nil, nil))
    let hScale = naturalWidth > 0 ? mappedQuad.width / naturalWidth : 1.0

    ctx.saveGState()
    ctx.translateBy(x: mappedQuad.bottomLeft.x, y: mappedQuad.bottomLeft.y)
    ctx.rotate(by: mappedQuad.rotationAngle)
    ctx.scaleBy(x: hScale, y: 1.0)
    ctx.textPosition = .zero
    CTLineDraw(ctLine, ctx)
    ctx.restoreGState()
}

func drawTextLayer(for result: PageResult, into ctx: CGContext, baseFont: CTFont) {
    ctx.setTextDrawingMode(.fill)

    for paragraph in result.paragraphs {
        guard !paragraph.lines.isEmpty else { continue }
        
        // 1. Zbieramy wysokości wszystkich linii w obrębie bieżącego akapitu
        let lineHeights = paragraph.lines.map { line -> CGFloat in
            return line.quad.mapped(using: result.geometry).height
        }
        
        // 2. Obliczamy medianę wysokości, aby odrzucić anomalie (np. indeksy górne, linie z samym łącznikiem)
        let sortedHeights = lineHeights.sorted()
        let medianHeight = sortedHeights[sortedHeights.count / 2]
        
        // 3. Obliczamy jeden ujednolicony rozmiar czcionki dla całego akapitu (zaokrąglony do 0.5 pkt)
        let rawParagraphSize = fontSizeForLineHeight(medianHeight, baseFont: baseFont)
        let paragraphFontSize = max(4.0, min(72.0, (rawParagraphSize * 2).rounded() / 2.0))
        
        // 4. Rysujemy wszystkie linie w akapicie, wymuszając ten sam rozmiar czcionki
        for line in paragraph.lines {
            guard !line.text.isEmpty else { continue }
            
            if !line.words.isEmpty {
                // Precyzyjne pozycjonowanie słowo po słowie
                for (index, word) in line.words.enumerated() {
                    guard !word.text.isEmpty else { continue }
                    
                    let isLast = (index == line.words.count - 1)
                    let textToDraw = isLast ? word.text : word.text + " "
                    
                    drawTextRun(
                        text: textToDraw,
                        quad: word.quad,
                        geometry: result.geometry,
                        ctx: ctx,
                        baseFont: baseFont,
                        fontSize: paragraphFontSize // Wymuszona czcionka akapitu
                    )
                }
            } else {
                // Bezpieczny fallback dla całych linii
                drawTextRun(
                    text: line.text,
                    quad: line.quad,
                    geometry: result.geometry,
                    ctx: ctx,
                    baseFont: baseFont,
                    fontSize: paragraphFontSize // Wymuszona czcionka akapitu
                )
            }
        }
    }
}

// MARK: - Zoptymalizowana kompresja tła do JPEG
func jpegCompressed(_ image: CGImage, quality: CGFloat) -> CGImage? {
    return autoreleasepool {
        let ciImage = CIImage(cgImage: image)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let options: [CIImageRepresentationOption: Any] = [
            CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality
        ]
        guard let data = sharedCIContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: options) else {
            return nil
        }
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(jpegDataProviderSource: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

// MARK: - Mapowanie metadanych PDFKit → klucze CGContext
func cgPDFContextMetadata(from attrs: [String: Any]) -> [String: Any] {
    let map: [(String, CFString)] = [
        ("Title",    kCGPDFContextTitle),
        ("Author",   kCGPDFContextAuthor),
        ("Subject",  kCGPDFContextSubject),
        ("Keywords", kCGPDFContextKeywords),
        ("Creator",  kCGPDFContextCreator),
    ]
    var out: [String: Any] = [:]
    for (pdfKey, cgKey) in map {
        if let val = attrs[pdfKey] { out[cgKey as String] = val }
    }
    return out
}

// MARK: - Zapis wynikowego PDF z precyzyjną obsługą rotacji i kompresji hybrydowej (PNG/JPEG/Lossless)
func writeOutputPDF(results: [PageResult], to outputURL: URL, metadata: [String: Any]? = nil, lossless: Bool = false) throws {
    let baseFont = CTFontCreateWithName("Helvetica" as CFString, 12, nil)
    let cgMeta = cgPDFContextMetadata(from: metadata ?? [:])

    guard let ctx = CGContext(outputURL as CFURL, mediaBox: nil, cgMeta as CFDictionary?) else {
        throw OCRPipelineError.contextCreationFailed
    }

    let sortedResults = results.sorted(by: { $0.logicalPageIndex < $1.logicalPageIndex })

    for part in sortedResults {
        autoreleasepool {
            let width = part.geometry.width
            let height = part.geometry.height
            var pageRect = CGRect(x: 0, y: 0, width: width, height: height)
            
            let boxData = Data(bytes: &pageRect, count: MemoryLayout<CGRect>.size) as CFData
            ctx.beginPDFPage([kCGPDFContextMediaBox: boxData] as CFDictionary)
            
            // 1. Wypełniamy całe płótno A4 białym tłem, aby ewentualne marginesy wyrównawcze stopiły się ze skanem
            ctx.setFillColor(NSColor.white.cgColor)
            ctx.fill(pageRect)
            
            // 2. Obliczamy pozycję wycentrowanego skanu książki na płótnie
            let centeredRect = CGRect(
                x: part.geometry.drawX,
                y: part.geometry.drawY,
                width: part.geometry.drawW,
                height: part.geometry.drawH
            )
            
            // 3. Rysujemy skan książki w wycentrowanej ramce
            if lossless {
                ctx.draw(part.backgroundImage, in: centeredRect)
            } else {
                let isGrayscale = part.backgroundImage.colorSpace?.model == .monochrome
                let quality: CGFloat = isGrayscale ? 0.78 : 0.82
                
                if let jpeg = jpegCompressed(part.backgroundImage, quality: quality) {
                    ctx.draw(jpeg, in: centeredRect)
                } else {
                    ctx.draw(part.backgroundImage, in: centeredRect)
                }
            }
            
            ctx.saveGState()
            drawTextLayer(for: part, into: ctx, baseFont: baseFont)
            ctx.restoreGState()
            ctx.endPDFPage()
        }
    }

    ctx.closePDF()
}
// MARK: - Output tekstowy
func joinParagraphText(_ lines: [RecognizedLine]) -> String {
    var result = ""
    for (i, line) in lines.enumerated() {
        var text = line.text
        let isLast = (i == lines.count - 1)

        if !isLast, text.hasSuffix("-"), let last = text.dropLast().last, last.isLetter {
            text.removeLast()
            result += text
        } else {
            result += text
            if !isLast { result += " " }
        }
    }
    return result
}

func writeSidecarText(results: [PageResult], to url: URL) throws {
    var output = ""
    for result in results {
        output += "=== Strona \(result.logicalPageIndex + 1) (z PDF org. \(result.originalPageIndex + 1)) ===\n\n"

        let bodyParas = result.paragraphs
            .filter { $0.role == .body }
            .sorted(by: { $0.readingOrderIndex < $1.readingOrderIndex })

        let footnoteParas = result.paragraphs
            .filter { $0.role == .footnote }
            .sorted(by: { $0.readingOrderIndex < $1.readingOrderIndex })

        for paragraph in bodyParas {
            output += joinParagraphText(paragraph.lines) + "\n\n"
        }
        
        if !footnoteParas.isEmpty {
            output += "--- Przypisy ---\n\n"
            for paragraph in footnoteParas {
                output += joinParagraphText(paragraph.lines) + "\n\n"
            }
        }
    }
    try output.write(to: url, atomically: true, encoding: .utf8)
}

// MARK: - [ML] Ochrona nazw własnych
func extractProtectedProperNouns(from text: String, language: String) -> Set<String> {
    let tagger = NLTagger(tagSchemes: [.nameType, .lexicalClass])
    tagger.string = text
    
    let lang = NLLanguage(rawValue: language.hasPrefix("pl") ? "pl" : language)
    tagger.setLanguage(lang, range: text.startIndex..<text.endIndex)

    var protected = Set<String>()

    tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word,
                          scheme: .nameType,
                          options: [.omitWhitespace, .omitPunctuation, .joinNames]) { tag, range in
        if tag == .personalName || tag == .placeName || tag == .organizationName {
            protected.insert(String(text[range]))
        }
        return true
    }

    tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word,
                          scheme: .lexicalClass,
                          options: [.omitWhitespace, .omitPunctuation]) { tag, range in
        let word = String(text[range])
        if tag == .noun, let first = word.first, first.isUppercase, word.count > 2 {
            protected.insert(word)
        }
        return true
    }

    return protected
}

private func cosineSimilarityVectors(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var dot = 0.0, normA = 0.0, normB = 0.0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    guard normA > 0, normB > 0 else { return 0 }
    return dot / (sqrt(normA) * sqrt(normB))
}

// MARK: - [ML] Zoptymalizowany, wątkowo bezpieczny silnik korekty tekstu
final class TextCorrectionEngine {
    private let spellChecker = NSSpellChecker.shared
    private let lock = NSRecursiveLock()
    
    private var validWordCache: [String: Bool] = [:]
    private var diacriticFixCache: [String: String] = [:]
    private var vectorCache: [String: [Double]] = [:]
    // Globalny akumulator nazwisk wyekstrahowanych z całej książki
    private(set) var accumulatedNames = Set<String>()
    
    private let useContextualNLP: Bool
    private var contextualEmbedding: Any? = nil
    
    // Klucz języka polskiego pomyślnie zwalidowany w systemie macOS
    private var resolvedLanguageKey = "pl"
    
    private static let confusionMap: [Character: [Character: Double]] = [
        "a": ["ą": 0.1, "o": 0.4],
        "ą": ["a": 0.1, "ę": 0.3, "o": 0.3, "ó": 0.4],
        "b": ["d": 0.5, "p": 0.4, "8": 0.3, "6": 0.3],
        "c": ["ć": 0.1, "o": 0.4, "e": 0.4, "s": 0.4],
        "ć": ["c": 0.1, "ś": 0.3],
        "d": ["z": 0.4, "o": 0.5, "b": 0.5],
        "e": ["ę": 0.1, "o": 0.5, "c": 0.4, "3": 0.4],
        "ę": ["e": 0.1, "ą": 0.3, "o": 0.3, "ó": 0.4],
        "g": ["q": 0.3, "9": 0.2],
        "h": ["k": 0.4],
        "i": ["t": 0.4, "j": 0.2, "l": 0.2, "1": 0.2],
        "j": ["ł": 0.3, "i": 0.2, "l": 0.2, "1": 0.2],
        "k": ["h": 0.4, "m": 0.5],
        "l": ["ł": 0.1, "i": 0.2, "1": 0.1, "j": 0.2, "t": 0.3],
        "ł": ["j": 0.3, "l": 0.1, "t": 0.4],
        "m": ["n": 0.4, "k": 0.5],
        "n": ["p": 0.5, "u": 0.3, "r": 0.4, "m": 0.4],
        "o": [
            "ó": 0.1, "a": 0.4, "e": 0.5, "u": 0.5, "d": 0.5, "p": 0.5, "c": 0.4, "ą": 0.3, "ę": 0.3, "0": 0.1
        ],
        "ó": [
            "o": 0.1, "u": 0.3, "ą": 0.4, "ę": 0.4, "0": 0.2
        ],
        "p": ["n": 0.5, "b": 0.4, "o": 0.5],
        "q": ["g": 0.3, "9": 0.3],
        "r": ["n": 0.4, "t": 0.4],
        "s": ["ś": 0.1, "z": 0.4, "c": 0.4, "5": 0.3],
        "ś": [
            "s": 0.1, "ć": 0.3, "ź": 0.4, "ż": 0.4
        ],
        "t": [
            "z": 0.3, "ł": 0.4, "l": 0.3, "i": 0.4, "r": 0.4, "1": 0.3, "7": 0.3
        ],
        "u": ["n": 0.3, "y": 0.4, "o": 0.5, "ó": 0.3],
        "v": ["w": 0.2],
        "w": ["v": 0.2],
        "y": ["u": 0.4],
        "z": [
            "t": 0.3, "d": 0.4, "ź": 0.1, "ż": 0.1, "s": 0.4, "2": 0.3
        ],
        "ź": [
            "z": 0.1, "ż": 0.2, "ś": 0.4
        ],
        "ż": [
            "z": 0.1, "ź": 0.2, "ś": 0.4
        ],
        "0": [
            "o": 0.1, "ó": 0.2
        ],
        "1": [
            "j": 0.2, "l": 0.1, "i": 0.2, "t": 0.3
        ],
        "2": [
            "z": 0.3
        ],
        "3": [
            "8": 0.3, "e": 0.4
        ],
        "5": [
            "s": 0.3
        ],
        "6": [
            "b": 0.3
        ],
        "7": [
            "t": 0.3
        ],
        "8": [
            "b": 0.3, "3": 0.3
        ],
        "9": [
            "g": 0.2, "q": 0.3
        ]
    ]
    
    private static let bigramConfusions: [(String, String, Double)] = [
        ("rn", "m", 0.1), ("m", "rn", 0.1),
        ("nn", "m", 0.2), ("m", "nn", 0.2),
        ("ri", "n", 0.2), ("n", "ri", 0.2),
        ("ii", "n", 0.2), ("n", "ii", 0.2),
        ("in", "m", 0.2), ("m", "in", 0.2),
        ("ni", "m", 0.3), ("m", "ni", 0.3),
        ("il", "h", 0.2), ("h", "il", 0.2),
        ("li", "h", 0.2), ("h", "li", 0.2),
        ("ll", "h", 0.2), ("h", "ll", 0.2),
        ("lt", "h", 0.3), ("h", "lt", 0.3),
        ("tl", "h", 0.3), ("h", "tl", 0.3),
        ("w", "vv", 0.1), ("vv", "w", 0.1),
        ("w", "iv", 0.2), ("iv", "w", 0.2),
        ("w", "vi", 0.2), ("vi", "w", 0.2),
        ("d", "cl", 0.2), ("cl", "d", 0.2),
        ("d", "ci", 0.2), ("ci", "d", 0.2),
        ("d", "ol", 0.3), ("ol", "d", 0.3),
        ("ń", "ni", 0.1), ("ni", "ń", 0.1),
        ("ń", "ii", 0.2), ("ii", "ń", 0.2),
        ("ń", "in", 0.2), ("in", "ń", 0.2),
        ("ł", "lt", 0.3), ("lt", "ł", 0.3),
        ("ł", "tl", 0.3), ("tl", "ł", 0.3),
        ("ł", "lj", 0.3), ("lj", "ł", 0.3),
        ("ł", "ij", 0.3), ("ij", "ł", 0.3),
        ("ą", "a,", 0.2), ("a,", "ą", 0.2),
        ("ę", "e,", 0.2), ("e,", "ę", 0.2),
        ("ﬁ", "fi", 0.1), ("fi", "ﬁ", 0.1),
        ("ﬂ", "fl", 0.1), ("fl", "ﬂ", 0.1),
        ("ﬀ", "ff", 0.1), ("ff", "ﬀ", 0.1),
        ("ﬃ", "ffi", 0.1), ("ffi", "ﬃ", 0.1),
        ("ﬄ", "ffl", 0.1), ("ffl", "ﬄ", 0.1),
        ("rz", "ż", 0.1), ("ż", "rz", 0.1),
        ("ch", "h", 0.1), ("h", "ch", 0.1),
        ("u", "ó", 0.1), ("ó", "u", 0.1),
        ("cz", "ć", 0.3), ("ć", "cz", 0.3),
        ("sz", "ś", 0.3), ("ś", "sz", 0.3),
        ("zi", "ź", 0.3), ("ź", "zi", 0.3),
        ("rz", "ź", 0.4), ("ź", "rz", 0.4),
        ("av", "aw", 0.3), ("aw", "av", 0.3),
        ("vv", "w", 0.1), ("w", "vv", 0.1),
        ("cl", "ol", 0.3), ("ol", "cl", 0.3),
        ("1i", "ł", 0.4), ("ł", "1i", 0.4),
        ("l1", "ł", 0.4), ("ł", "l1", 0.4),
        ("11", "ł", 0.4), ("ł", "11", 0.4),
        ("0o", "ó", 0.4), ("ó", "0o", 0.4)
    ]
    
    private func isSuspiciousCharacter(_ char: Character) -> Bool {
        if char.isNumber { return true }
        let charStr = String(char)
        let suspiciousStrings: Set<String> = ["ı", "ﬁ", "ﬂ"]
        return suspiciousStrings.contains(charStr)
    }
    
    private func hardCleanReplacements(_ word: String) -> String {
        let lower = word.lowercased()
        let rules: [String: String] = [
            "pśychol1zycznego": "psychofizycznego",
            "0be1-":            "obej-",
            "c2e1":             "czej",
            "litymı":           "fizyki",
            "staroyvinej":      "starożytnej",
            "1i":               "ł",
            "pęźnaw-":          "poznaw-",
            "mierzalnośé":      "mierzalność",
            "padzią-":          "podzia-",
            "wyślępulą":        "występują",
            "wujosków":         "wniosków",
            "indornacji":       "informacji",
            "poszumiwoniem":    "poszukiwaniem",
        ]
        if let matched = rules[lower] {
            return word.first?.isUppercase == true ? matched.capitalized : matched
        }
        return word
    }
    
    private func weightedLevenshtein(_ s1: String, _ s2: String) -> Double {
        let u1 = Array(s1.lowercased())
        let u2 = Array(s2.lowercased())
        let m = u1.count
        let n = u2.count
        
        var dp = [[Double]](repeating: [Double](repeating: 0.0, count: n + 1), count: m + 1)
        
        for i in 0...m { dp[i][0] = Double(i) }
        for j in 0...n { dp[0][j] = Double(j) }
        
        for i in 1...m {
            for j in 1...n {
                let char1 = u1[i - 1]
                let char2 = u2[j - 1]
                
                let insertionCost = 1.0
                let deletionCost = 1.0
                
                var substitutionCost = 1.0
                if char1 == char2 {
                    substitutionCost = 0.0
                } else if let cost = TextCorrectionEngine.confusionCost(char1, char2) {
                    substitutionCost = cost
                }
                
                dp[i][j] = min(
                    dp[i - 1][j] + deletionCost,
                    dp[i][j - 1] + insertionCost,
                    dp[i - 1][j - 1] + substitutionCost
                )
            }
        }
        return dp[m][n]
    }
    
    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let empty = [Int](repeating: 0, count: s2.count + 1)
        var d = [[Int]](repeating: empty, count: s1.count + 1)
        
        for i in 0...s1.count { d[i][0] = i }
        for j in 0...s2.count { d[0][j] = j }
        
        let a1 = Array(s1)
        let a2 = Array(s2)
        
        if s1.count == 0 { return s2.count }
        if s2.count == 0 { return s1.count }
        
        for i in 1...s1.count {
            for j in 1...s2.count {
                if a1[i-1] == a2[j-1] {
                    d[i][j] = d[i-1][j-1]
                } else {
                    d[i][j] = min(
                        d[i-1][j] + 1,
                        d[i][j-1] + 1,
                        d[i-1][j-1] + 1
                    )
                }
            }
        }
        return d[s1.count][s2.count]
    }
    
    init(useContextualNLP: Bool) {
        self.useContextualNLP = useContextualNLP
        if useContextualNLP {
            if #available(macOS 14.0, *) {
                self.contextualEmbedding = NLContextualEmbedding(language: .polish)
            }
        }
        
        // NIEOSZUKIWALNA PROCEDURA AUTODIAGNOSTYCZNA (Test słowa "książka")
        let testWord = "książka"
        let keysToTest = ["pl", "pl_PL", "Polish"]
        var dictionaryLoaded = false
        
        for key in keysToTest {
            if spellChecker.setLanguage(key) {
                let range = spellChecker.checkSpelling(
                    of: testWord, startingAt: 0, language: key, wrap: false,
                    inSpellDocumentWithTag: 0, wordCount: nil
                )
                if range.location == NSNotFound {
                    self.resolvedLanguageKey = key
                    dictionaryLoaded = true
                    print("🎯 NSSpellChecker: Pomyślnie zweryfikowano i aktywowano prawdziwy słownik języka polskiego ('\(key)')")
                    break
                }
            }
        }
        
        if !dictionaryLoaded {
            print("\n⚠️ OSTRZEŻENIE: Prawdziwy słownik języka polskiego (NSSpellChecker) NIE JEST aktywny na Twoim Macu.")
            print("   Obecnie system używa słownika zastępczego (np. angielskiego), co uniemożliwia korektę.")
            print("   Aby to naprawić, wejdź w: Ustawienia Systemowe -> Klawiatura -> Pisownia i ustaw na 'Polski'.\n")
        }
    }
    
    func prepareEmbedding() async {
        guard useContextualNLP else { return }
        if #available(macOS 14.0, *) {
            guard let embedding = contextualEmbedding as? NLContextualEmbedding else { return }
            if !embedding.hasAvailableAssets {
                print("🌐 Pobieranie systemowych zasobów NLP (BERT) dla języka polskiego...")
                do {
                    let result = try await embedding.requestAssets()
                    guard result == .available else {
                        print("⚠️ Systemowe zasoby NLP są niedostępne.")
                        return
                    }
                    print("✅ Zasoby NLP pobrane pomyślnie.")
                } catch {
                    print("❌ Błąd pobierania zasobów NLP: \(error)")
                    return
                }
            }
            
            do {
                try embedding.load()
                print("🧠 Model osadzeń kontekstowych (BERT) został pomyślnie wczytany.")
            } catch {
                print("❌ Nie udało się załadować modelu NLP: \(error)")
            }
        }
    }
    
    private func getContextualVector(for text: String) -> [Double]? {
        lock.lock()
        defer { lock.unlock() }
        
        if let cached = vectorCache[text] {
            return cached
        }
        
        if #available(macOS 14.0, *), useContextualNLP {
            guard let embedding = contextualEmbedding as? NLContextualEmbedding, embedding.hasAvailableAssets else { return nil }
            guard let result = try? embedding.embeddingResult(for: text, language: .polish) else { return nil }
            
            let dim = embedding.dimension
            var sumVector = [Double](repeating: 0.0, count: dim)
            var tokenCount = 0
            
            let range = text.startIndex..<text.endIndex
            result.enumerateTokenVectors(in: range) { (vector, tokenRange) -> Bool in
                for (offset, val) in vector.enumerated() {
                    sumVector[offset] += val
                }
                tokenCount += 1
                return true
            }
            
            guard tokenCount > 0 else { return nil }
            let finalVector = sumVector.map { $0 / Double(tokenCount) }
            
            vectorCache[text] = finalVector
            return finalVector
        }
        return nil
    }
    
    private func getContextualizedWordVector(in sentence: String, wordRange: Range<String.Index>) -> [Double]? {
        guard #available(macOS 14.0, *), useContextualNLP else { return nil }
        guard let embedding = contextualEmbedding as? NLContextualEmbedding, embedding.hasAvailableAssets else { return nil }
        guard let result = try? embedding.embeddingResult(for: sentence, language: .polish) else { return nil }
        
        let dim = embedding.dimension
        var sumVector = [Double](repeating: 0.0, count: dim)
        var tokenCount = 0
        
        result.enumerateTokenVectors(in: wordRange) { (vector, tokenRange) -> Bool in
            for (offset, val) in vector.enumerated() {
                sumVector[offset] += val
            }
            tokenCount += 1
            return true
        }
        
        guard tokenCount > 0 else { return nil }
        return sumVector.map { $0 / Double(tokenCount) }
    }
    
    private func generateOpticalCandidates(for core: String, language: String) -> [String] {
        var candidates = Set<String>()
        
        for entry in TextCorrectionEngine.bigramConfusions {
            let src = entry.0
            let dst = entry.1
            
            var searchStart = core.startIndex
            while let range = core.range(of: src, options: .caseInsensitive, range: searchStart..<core.endIndex) {
                var variant = core
                
                let originalSegment = variant[range]
                var finalDst = dst
                if let firstChar = originalSegment.first, firstChar.isUppercase {
                    finalDst = dst.capitalized
                }
                
                variant.replaceSubrange(range, with: finalDst)
                
                if isKnownValidWord(variant, language: language) {
                    candidates.insert(variant)
                }
                
                searchStart = range.upperBound
                if candidates.count >= 20 { break }
            }
            if candidates.count >= 20 { break }
        }
        
        struct Substitution {
            let index: Int
            let original: Character
            let replacement: Character
            let cost: Double
        }
        
        let chars = Array(core)
        var availableSubstitutions: [Substitution] = []
        for (idx, char) in chars.enumerated() {
            let lowerChar = Character(char.lowercased())
            if let targets = TextCorrectionEngine.confusionMap[lowerChar] {
                for (rep, cost) in targets {
                    let finalRep = char.isUppercase ? Character(rep.uppercased()) : rep
                    availableSubstitutions.append(Substitution(index: idx, original: char, replacement: finalRep, cost: cost))
                }
            }
        }
        
        let sortedSubs = availableSubstitutions.sorted { $0.cost < $1.cost }
        let topSubs = Array(sortedSubs.prefix(25))
        
        for sub in topSubs {
            var variant = chars
            variant[sub.index] = sub.replacement
            let variantStr = String(variant)
            if isKnownValidWord(variantStr, language: language) {
                candidates.insert(variantStr)
            }
            if candidates.count >= 30 { break }
        }
        
        if topSubs.count > 1 && candidates.count < 30 {
            for i in 0..<topSubs.count {
                for j in (i + 1)..<topSubs.count {
                    let sub1 = topSubs[i]
                    let sub2 = topSubs[j]
                    guard sub1.index != sub2.index else { continue }
                    
                    var variant = chars
                    variant[sub1.index] = sub1.replacement
                    variant[sub2.index] = sub2.replacement
                    let variantStr = String(variant)
                    if isKnownValidWord(variantStr, language: language) {
                        candidates.insert(variantStr)
                    }
                    if candidates.count >= 30 { break }
                }
                if candidates.count >= 30 { break }
            }
        }
        
        return Array(candidates)
    }
    
    func pickBestSpellingCandidate(_ candidates: [String], originalCore: String, contextBefore: [String], contextAfter: [String]) -> String? {
            guard candidates.count > 1 else { return candidates.first }
            
            // 1. Obliczamy odległości Levenshteina dla wszystkich kandydatów
            let candidatesWithDistances = candidates.map { candidate -> (word: String, distance: Double) in
                return (candidate, self.weightedLevenshtein(originalCore, candidate))
            }
            
            // 2. Znajdujemy minimalną odległość edycyjną
            let minDistance = candidatesWithDistances.map { $0.distance }.min() ?? 99.0
            
            // 3. Zostawiamy TYLKO kandydatów o najlepszym, identycznym dopasowaniu optycznym (remis)
            let bestOpticalCandidates = candidatesWithDistances.filter { abs($0.distance - minDistance) < 0.01 }.map { $0.word }
            
            // Jeśli jest tylko jeden najlepszy kandydat optyczny, zwracamy go natychmiast bez angażowania NLP!
            if bestOpticalCandidates.count == 1 {
                return bestOpticalCandidates[0]
            }
            
            // 4. ROZSTRZYGNIĘCIE REMISU: Ścieżka A (Contextual Sentence Substitution przy użyciu systemowego BERT)
            if #available(macOS 14.0, *), useContextualNLP, contextualEmbedding != nil {
                let contextSentence = (contextBefore + contextAfter).joined(separator: " ")
                guard !contextSentence.isEmpty, let contextVector = getContextualVector(for: contextSentence) else {
                    return pickBestSpellingCandidateLegacy(bestOpticalCandidates, originalCore: originalCore, context: contextBefore + contextAfter)
                }
                
                var best: (word: String, score: Double)?
                for candidate in bestOpticalCandidates {
                    // Konstruujemy zdanie próbne bezpieczną metodą
                    let prefix = contextBefore.joined(separator: " ") + (contextBefore.isEmpty ? "" : " ")
                    let suffix = (contextAfter.isEmpty ? "" : " ") + contextAfter.joined(separator: " ")
                    let sentence = prefix + candidate + suffix
                    
                    // Obliczamy bezpieczny punkt startu wyszukiwania kandydata
                    let searchStart = sentence.index(sentence.startIndex, offsetBy: prefix.count)
                    
                    // Szukamy kandydata wewnątrz zdania przy użyciu bezpiecznych funkcji systemowych.
                    // Chroni to przed wszelkimi anomaliami długości znaków Unicode.
                    guard let candidateRange = sentence.range(of: candidate, range: searchStart..<sentence.endIndex) else {
                        continue
                    }
                    
                    // Pobieramy wektor kandydata uwarunkowany kontekstem tego zdania (BERT)
                    guard let contextualizedVector = getContextualizedWordVector(in: sentence, wordRange: candidateRange) else { continue }
                    
                    // Porównujemy wektor kandydata z wektorem otaczającego kontekstu
                    let cosine = cosineSimilarityVectors(contextVector, contextualizedVector)
                    
                    if best == nil || cosine > best!.score {
                        best = (candidate, cosine)
                    }
                }
                
                return best?.word ?? pickBestSpellingCandidateLegacy(bestOpticalCandidates, originalCore: originalCore, context: contextBefore + contextAfter)
            }
            
            return pickBestSpellingCandidateLegacy(bestOpticalCandidates, originalCore: originalCore, context: contextBefore + contextAfter)
        }
    
    private func pickBestSpellingCandidateLegacy(_ candidates: [String], originalCore: String, context: [String]) -> String? {
        let candidatesWithDistances = candidates.map { (word: $0, distance: self.weightedLevenshtein(originalCore, $0)) }
        let minDistance = candidatesWithDistances.map { $0.distance }.min() ?? 99.0
        let bestOptical = candidatesWithDistances.filter { abs($0.distance - minDistance) < 0.01 }.map { $0.word }
        
        if bestOptical.count == 1 {
            return bestOptical[0]
        }
        
        guard let embedding = NLEmbedding.wordEmbedding(for: .polish) else {
            return bestOptical.first
        }
        
        let contextVectors: [[Double]] = context.compactMap { embedding.vector(for: $0.lowercased()) }
        guard !contextVectors.isEmpty else {
            return bestOptical.first
        }
        
        let dim = contextVectors[0].count
        var avgContext = [Double](repeating: 0, count: dim)
        for vec in contextVectors {
            for i in 0..<dim { avgContext[i] += vec[i] }
        }
        for i in 0..<dim { avgContext[i] /= Double(contextVectors.count) }
        
        var best: (word: String, score: Double)?
        for candidate in bestOptical {
            guard let vec = embedding.vector(for: candidate.lowercased()) else { continue }
            let cosine = cosineSimilarityVectors(avgContext, vec)
            
            if best == nil || cosine > best!.score {
                best = (candidate, cosine)
            }
        }
        return best?.word ?? bestOptical.first
    }
    
    private func stripPolishDiacritics(_ s: String) -> String {
        var temp = s
            .replacingOccurrences(of: "ł", with: "l")
            .replacingOccurrences(of: "Ł", with: "L")
        return temp.folding(options: .diacriticInsensitive, locale: Locale(identifier: "pl_PL"))
    }
    
    private func detectLanguage(for text: String) -> String {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        if let language = recognizer.dominantLanguage {
            return language.rawValue
        }
        return "pl"
    }
    
    private func buildProtectedWords(text: String, customWords: [String]) -> Set<String> {
        var protected = Set<String>()
        protected.formUnion(customWords)

        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text
        let options: NLTagger.Options = [.omitWhitespace, .omitPunctuation, .joinNames]

        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType, options: options) { tag, range in
            guard let tag = tag else { return true }
            switch tag {
            case .personalName, .placeName, .organizationName:
                let entity = String(text[range])
                protected.insert(entity)
            default:
                break
            }
            return true
        }

        let patterns = [
            #"\(([A-ZŁŚŻŹĆŃÓĄĘ][A-Za-zÀ-ÿŁŚŻŹĆŃÓĄĘłśżźćńóąę\-]+)\s*,?\s*\d{4}\)"#,
            #"\[([A-ZŁŚŻŹĆŃÓĄĘ][A-Za-zÀ-ÿŁŚŻŹĆŃÓĄĘłśżźćńóąę\-]+)\s+\d{4}\]"#
        ]

        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let nsRange = NSRange(text.startIndex..., in: text)

            regex.enumerateMatches(in: text, options: [], range: nsRange) { match, _, _ in
                guard let match = match,
                      match.numberOfRanges > 1,
                      let range = Range(match.range(at: 1), in: text) else { return }
                
                protected.insert(String(text[range]))
            }
        }

        return protected
    }
    
    func correctSinglePage(_ page: PageResult, fixDiacritics: Bool, splitMergedWords: Bool, customWords: [String] = [], defaultLanguage: String = "pl") -> PageResult {
            let activeLanguage = defaultLanguage
            
            let fullPageText = page.paragraphs.flatMap { $0.lines.map { $0.text } }.joined(separator: " ")
            let protectedWords = buildProtectedWords(text: fullPageText, customWords: customWords)
            
            lock.lock()
            // Akumulujemy wykryte nazwiska w globalnym zbiorze klasy
            self.accumulatedNames.formUnion(protectedWords)
            lock.unlock()
            
            let newParagraphs = page.paragraphs.map { paragraph -> Paragraph in
                var p = paragraph
                var correctedLines: [RecognizedLine] = []
                
                var linesTokens: [[String]] = paragraph.lines.map { line in
                    line.text.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
                }
            
                // Rekonstrukcja słów przenoszonych łącznikiem między wierszami
                if linesTokens.count > 1 {
                    for i in 0..<(linesTokens.count - 1) {
                        if let lastToken = linesTokens[i].last, lastToken.hasSuffix("-"), !linesTokens[i+1].isEmpty {
                            let firstToken = linesTokens[i+1][0]
                            let mergedWord = String(lastToken.dropLast()) + firstToken
                            
                            let correctedMerged = self.correctToken(mergedWord, language: activeLanguage, fixDiacritics: fixDiacritics,
                                                                    splitMergedWords: splitMergedWords, protectedWords: protectedWords,
                                                                    contextBefore: [], contextAfter: [])
                            
                            let cutIndex = lastToken.count - 1
                            if correctedMerged.count >= cutIndex {
                                let index = correctedMerged.index(correctedMerged.startIndex, offsetBy: cutIndex)
                                let part1 = String(correctedMerged[..<index]) + "-"
                                let part2 = String(correctedMerged[index...])
                                
                                linesTokens[i][linesTokens[i].count - 1] = part1
                                linesTokens[i+1][0] = part2
                            }
                        }
                    }
                }
                
                // Korekta pozostałych słów
                for (lineIdx, tokens) in linesTokens.enumerated() {
                    let line = paragraph.lines[lineIdx]
                    let correctedTokens = tokens.enumerated().map { idx, tok -> String in
                        let start = max(0, idx - 2)
                        
                        // Bezpieczne, odporne na crashe wycinanie przedziałów tablicy słów (kontekst przed i po)
                        let contextBefore = Array(tokens[start..<idx])
                        let contextAfter = (idx + 1) < tokens.count ? Array(tokens[(idx + 1)..<min(tokens.count, idx + 3)]) : []
                        
                        return self.correctToken(tok, language: activeLanguage, fixDiacritics: fixDiacritics,
                                                 splitMergedWords: splitMergedWords, protectedWords: protectedWords,
                                                 contextBefore: contextBefore, contextAfter: contextAfter)
                    }
                    
                    let newText = correctedTokens.joined(separator: " ")
                    let newWords = line.words.map { word -> RecognizedWord in
                        let correctedWord = self.correctToken(word.text, language: activeLanguage, fixDiacritics: fixDiacritics,
                                                              splitMergedWords: splitMergedWords, protectedWords: protectedWords,
                                                              contextBefore: [], contextAfter: [])
                        return RecognizedWord(text: correctedWord, quad: word.quad)
                    }
                    
                    correctedLines.append(RecognizedLine(
                        text: newText,
                        quad: line.quad,
                        confidence: line.confidence,
                        words: newWords,
                        role: line.role
                    ))
                }
                
                p.lines = correctedLines
                return p
            }
            
            return PageResult(
                logicalPageIndex: page.logicalPageIndex,
                originalPageIndex: page.originalPageIndex,
                geometry: page.geometry,
                backgroundImage: page.backgroundImage,
                paragraphs: newParagraphs,
                originalPDFPage: page.originalPDFPage
            )
        }
    
    private func correctToken(_ token: String, language: String, fixDiacritics: Bool, splitMergedWords: Bool, protectedWords: Set<String>, contextBefore: [String], contextAfter: [String]) -> String {
        guard let firstLetterIdx = token.firstIndex(where: { $0.isLetter }),
              let lastLetterIdx = token.lastIndex(where: { $0.isLetter }) else {
            return token
        }
        
        let prefix = String(token[token.startIndex..<firstLetterIdx])
        let suffixStart = token.index(after: lastLetterIdx)
        let suffix = suffixStart <= token.endIndex ? String(token[suffixStart...]) : ""
        let core = String(token[firstLetterIdx...lastLetterIdx])
        
        if protectedWords.contains(core) {
            return token
        }
        
        for protected in protectedWords {
            if weightedLevenshtein(core.lowercased(), protected.lowercased()) == 1.0 {
                let isCapitalized = core.first?.isUppercase ?? false
                let corrected = isCapitalized ? protected.capitalized : protected.lowercased()
                return prefix + corrected + suffix
            }
        }
        
        var modifiedCore = hardCleanReplacements(core)
        
        if fixDiacritics {
            modifiedCore = applyDiacriticFix(modifiedCore, language: language, contextBefore: contextBefore, contextAfter: contextAfter)
        }
        
        if splitMergedWords && modifiedCore.count >= OCRTuning.mergedWordMinLength && modifiedCore.count <= OCRTuning.mergedWordMaxTotalLength {
            if let splitted = applySplitMerged(modifiedCore, language: language) {
                modifiedCore = splitted
            }
        }
        
        modifiedCore = hardCleanReplacements(modifiedCore)
        return prefix + modifiedCore + suffix
    }
    
    private func applyDiacriticFix(_ core: String, language: String, contextBefore: [String], contextAfter: [String]) -> String {
        lock.lock()
        defer { lock.unlock() }
        
        if let cached = diacriticFixCache[core] {
            return cached
        }
        
        var normalizedCore = core
        let hasSuspicious = core.contains(where: { isSuspiciousCharacter($0) })
        if hasSuspicious {
            var chars = Array(core)
            for idx in 0..<chars.count {
                if chars[idx] == "1" {
                    chars[idx] = "i"
                } else if chars[idx] == "0" {
                    chars[idx] = "o"
                } else if chars[idx] == "8" {
                    chars[idx] = "b"
                } else if chars[idx] == "3" {
                    chars[idx] = "e"
                } else if chars[idx] == "5" {
                    chars[idx] = "s"
                } else if chars[idx] == "6" {
                    chars[idx] = "b"
                } else if chars[idx] == "9" {
                    chars[idx] = "g"
                } else if chars[idx] == "2" {
                    chars[idx] = "z"
                } else if chars[idx] == "4" {
                    chars[idx] = "a"
                }
            }
            normalizedCore = String(chars)
        }
        
        guard normalizedCore.count >= 3,
              normalizedCore.range(of: "^[0-9]+$", options: .regularExpression) == nil,
              normalizedCore != normalizedCore.uppercased() else {
            return core
        }
        
        let isPolish = (language == "pl" || language == "pl_PL" || language == "Polish")
        let activeLanguage = isPolish ? resolvedLanguageKey : language
        
        if isKnownValidWord(normalizedCore, language: activeLanguage) { return normalizedCore }
        
        var standardGuesses = Set<String>()
        
        _ = spellChecker.setLanguage(activeLanguage)
        
        let spellGuesses = spellChecker.guesses(
            forWordRange: NSRange(location: 0, length: (normalizedCore as NSString).length),
            in: normalizedCore, language: activeLanguage, inSpellDocumentWithTag: 0
        )
        
        if let guesses = spellGuesses {
            standardGuesses.formUnion(guesses)
        }
        
        let opticalCandidates = generateOpticalCandidates(for: normalizedCore, language: activeLanguage)
        let strippedOriginal = stripPolishDiacritics(normalizedCore).lowercased()
        
        let filteredStandard = standardGuesses.filter { guess in
            let strippedGuess = stripPolishDiacritics(guess).lowercased()
            return guess != normalizedCore && strippedGuess == strippedOriginal
        }
        
        var finalCandidatesSet = Set<String>(filteredStandard)
        finalCandidatesSet.formUnion(opticalCandidates)
        
        let filteredCandidates = Array(finalCandidatesSet)
        
        var result = normalizedCore
        if filteredCandidates.count == 1 {
            result = filteredCandidates[0]
        } else if filteredCandidates.count > 1, let picked = pickBestSpellingCandidate(filteredCandidates, originalCore: normalizedCore, contextBefore: contextBefore, contextAfter: contextAfter) {
            result = picked
        }
        
        diacriticFixCache[core] = result
        return result
    }
    
    private func isKnownValidWord(_ w: String, language: String) -> Bool {
        if w.contains(where: { $0.isNumber }) && w.contains(where: { $0.isLetter }) {
            return false
        }
        
        let lower = w.lowercased()
        
        lock.lock()
        defer { lock.unlock() }
        
        if let cached = validWordCache[lower] {
            return cached
        }
        
        let result: Bool
        if w.count == 1 {
            if language == "pl" || language == "pl_PL" || language == "Polish" {
                result = ["a", "i", "o", "u", "w", "z"].contains(lower)
            } else if language == "en" {
                result = ["a", "i"].contains(lower)
            } else {
                result = true
            }
        } else {
            let isPolish = (language == "pl" || language == "pl_PL" || language == "Polish")
            let activeLangKey = isPolish ? resolvedLanguageKey : language
            
            _ = spellChecker.setLanguage(activeLangKey)
            
            let range = spellChecker.checkSpelling(
                of: w, startingAt: 0, language: activeLangKey, wrap: false,
                inSpellDocumentWithTag: 0, wordCount: nil
            )
            result = (range.location == NSNotFound)
        }
        
        validWordCache[lower] = result
        return result
    }
    
    private func applySplitMerged(_ core: String, language: String) -> String? {
        guard core.range(of: "[0-9]", options: .regularExpression) == nil,
              core != core.uppercased(),
              !isKnownValidWord(core, language: language) else { return nil }
        
        let chars = Array(core)
        let n = chars.count
        var memoSuccess: [Int: [String]] = [n: []]
        var failed: Set<Int> = []
        
        func solve(_ i: Int) -> [String]? {
            if let cached = memoSuccess[i] { return cached }
            if failed.contains(i) { return nil }
            
            let maxLen = min(n - i, OCRTuning.mergedWordMaxSegmentLength)
            for len in stride(from: maxLen, through: 1, by: -1) {
                let candidate = String(chars[i..<(i + len)])
                if isKnownValidWord(candidate, language: language),
                   let rest = solve(i + len) {
                    let result = [candidate] + rest
                    memoSuccess[i] = result
                    return result
                }
            }
            failed.insert(i)
            return nil
        }
        
        guard let segments = solve(0), segments.count >= 2 else { return nil }
        
        let avgLen = Double(core.count) / Double(segments.count)
        guard avgLen >= OCRTuning.minAvgSegmentLength else { return nil }
        
        return segments.joined(separator: " ")
    }
    
    private static func confusionCost(_ char1: Character, _ char2: Character) -> Double? {
        let c1 = Character(char1.lowercased())
        let c2 = Character(char2.lowercased())
        if c1 == c2 { return 0.0 }
        if let cost = confusionMap[c1]?[c2] { return cost }
        if let cost = confusionMap[c2]?[c1] { return cost }
        return nil
    }
}

// MARK: - Zintegrowany walidator wyjściowego PDF
final class PDFValidator {
    static func verifyPDF(at url: URL, expectedPageCount: Int) {
        print("🔍 Rozpoczynanie walidacji wyjściowego pliku PDF...")
        guard let document = PDFDocument(url: url) else {
            print("❌ Walidacja nieudana: Nie można załadować wygenerowanego pliku PDF.")
            return
        }

        let actualPageCount = document.pageCount
        if actualPageCount != expectedPageCount {
            print("⚠️ Ostrzeżenie walidacji: Niezgodność liczby stron. Oczekiwano: \(expectedPageCount), Wygenerowano: \(actualPageCount)")
        }

        var searchablePages = 0
        var totalCharacters = 0

        for i in 0..<actualPageCount {
            guard let page = document.page(at: i) else { continue }
            if let pageText = page.string, !pageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                searchablePages += 1
                totalCharacters += pageText.count
            }
        }

        let searchabilityRatio = Double(searchablePages) / Double(actualPageCount)
        print("📊 Raport Walidacji PDF:")
        print("   - Liczba stron z możliwością przeszukiwania: \(searchablePages)/\(actualPageCount) (\(String(format: "%.1f", searchabilityRatio * 100))%)")
        print("   - Całkowita liczba wyodrębnionych znaków: \(totalCharacters)")
        
        if searchablePages > 0 {
            print("✅ Walidacja zakończona pomyślnie. Warstwa tekstowa jest zaznaczalna i przeszukiwalna.")
        } else {
            print("❌ Krytyczne ostrzeżenie: Wygenerowany PDF nie zawiera przeszukiwalnej warstwy tekstowej.")
        }
    }
}

func parseArguments() -> Config {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        let helpText = "Użycie: swift ocr-nlp-bert-gemini.swift input.pdf output.pdf [opcje]\n\n" +
                       "Opcje:\n" +
                       "  --dpi <liczba>            rozdzielczość renderowania (domyślnie 300)\n" +
                       "  --lang <l1,l2,...>        języki OCR (domyślnie pl-PL)\n" +
                       "  --sidecar <plik.txt>      eksport tekstu logicznego w reading order\n" +
                       "  --customwords <a,b>       dodatkowe słownictwo dla Vision (np. nazwiska)\n" +
                       "  --nosplit                 zachowaj wygląd strony wejściowej (bez cięcia rozkładówek)\n" +
                       "  --color                   wymuś renderowanie tła w kolorze (sRGB) zamiast skali szarości\n" +
                       "\n" +
                       "  Jakość OCR / diakrytyki:\n" +
                       "  --upscale <liczba>        powiększenie obrazu przed OCR, np. 1.4 (domyślnie 1.0)\n" +
                       "  --despeckle               włącza odszumianie (median filter)\n" +
                       "  --fix-diacritics          post-OCR korekta brakujących znaków diakrytycznych\n" +
                       "                            (+ zamknięcie morfologiczne przed OCR)\n" +
                       "  --split-merged-words     post-OCR separacja zlepionych wyrazów (DP + NSSpellChecker)\n" +
                       "  --nlp                     włącza zaawansowaną korektę kontekstową (systemowy model BERT)\n" +
                       "\n" +
                       "  Bezpieczne kadrowanie:\n" +
                       "  --autocrop                włącza wykrywanie i przycinanie marginesów/cieni\n" +
                       "  --autocrop-padding <f>    bezpieczny margines wokół treści, fraction (domyślnie 0.015)\n" +
                       "  --autocrop-threshold <f>  minimalna redukcja, by kadrowanie się \"opłacało\" (domyślnie 0.02)\n"
        print(helpText)
        exit(1)
    }

    var config = Config(inputPath: args[1], outputPath: args[2])
    var i = 3
    while i < args.count {
        switch args[i] {
        case "--dpi":
            if i + 1 < args.count, let v = Double(args[i + 1]) { config.dpi = CGFloat(v) }
            i += 2
        case "--lang":
            if i + 1 < args.count {
                config.languages = args[i + 1].split(separator: ",").map(String.init)
            }
            i += 2
        case "--sidecar":
            if i + 1 < args.count { config.sidecarPath = args[i + 1] }
            i += 2
        case "--customwords":
                    if i + 1 < args.count {
                        let argument = args[i + 1]
                        // Sprawdzamy, czy podany parametr to ścieżka do pliku .txt
                        if argument.lowercased().hasSuffix(".txt") {
                            do {
                                let fileContent = try String(contentsOfFile: argument, encoding: .utf8)
                                // Wczytujemy słowa rozdzielone przecinkami, średnikami lub znakami nowej linii
                                config.customWords = fileContent
                                    .components(separatedBy: CharacterSet(charactersIn: ",\n\r;"))
                                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                                    .filter { !$0.isEmpty }
                                print("📖 Wczytano słownik niestandardowy z pliku: \(argument) (\(config.customWords.count) słów)")
                            } catch {
                                print("⚠️ Nie udało się odczytać pliku customwords: \(error.localizedDescription). Używam domyślnego słownika.")
                                config.customWords = []
                            }
                        } else {
                            // Jeśli to nie plik, parsujemy tradycyjnie jako ciąg tekstowy rozdzielony przecinkami
                            config.customWords = argument.split(separator: ",").map(String.init)
                        }
                    }
                    i += 2
            i += 2
        case "--upscale":
            if i + 1 < args.count, let v = Double(args[i + 1]) { OCRTuning.ocrUpscaleFactor = CGFloat(v) }
            i += 2
        case "--despeckle":
            OCRTuning.enableDespeckle = true
            i += 1
        case "--fix-diacritics":
            config.fixDiacritics = true
            i += 1
        case "--split-merged-words":
            config.splitMergedWords = true
            i += 1
        case "--nlp":
            config.useContextualNLP = true
            i += 1
        case "--autocrop":
            config.autocrop.enabled = true
            i += 1
        case "--autocrop-padding":
            if i + 1 < args.count, let v = Double(args[i + 1]) { config.autocrop.paddingFraction = CGFloat(v) }
            i += 2
        case "--autocrop-threshold":
            if i + 1 < args.count, let v = Double(args[i + 1]) { config.autocrop.brightnessThreshold = CGFloat(v) }
            i += 2
        case "--nosplit":
            config.noSplit = true
            i += 1
        case "--color":
                    config.color = true
                    i += 1
                case "--lossless":
                    config.lossless = true
                    i += 1
        default:
            i += 1
        }
    }
    return config
}

// MARK: - Aplikacja
func runPipeline() async throws {
    let config = parseArguments()
    let inputURL = URL(fileURLWithPath: config.inputPath)
    let outputURL = URL(fileURLWithPath: config.outputPath)

    guard let document = PDFDocument(url: inputURL) else {
        throw OCRPipelineError.cannotOpenDocument
    }

    var pdfMetadata: [String: Any] = [:]
    if let documentAttributes = document.documentAttributes {
        for (key, value) in documentAttributes {
            if let stringKey = key as? String {
                pdfMetadata[stringKey] = value
            } else if let pdfKey = key as? PDFDocumentAttribute {
                pdfMetadata[pdfKey.rawValue] = value
            }
        }
    }

    let pageCount = document.pageCount
    let scale = config.dpi / 72.0

    print("📄 \(config.inputPath) — Fizyczne strony: \(pageCount), DPI=\(Int(config.dpi)), języki=\(config.languages.joined(separator: ", "))")

    struct RenderTask {
        let logicalIndex: Int
        let originalIndex: Int
        let image: CGImage
        let geometry: PageGeometry
        let pdfPage: CGPDFPage?
    }

    var renderedPages: [RenderTask] = []
    var currentLogicalIndex = 0

    // Pętla renderująca w runPipeline() z przekazaniem flagi kolor:
    for i in 0..<pageCount {
        guard let page = document.page(at: i) else { continue }
        if config.noSplit {
            let (img, _, cgPage) = try renderPageNoSplit(page, scale: scale, color: config.color)
            
            // Programistyczna detekcja i pionowanie dla trybu --nosplit
            let (uprightImg, wasRotated, _) = await detectTrueTextOrientation(for: img, languages: config.languages)
            let uprightGeom = PageGeometry(width: CGFloat(uprightImg.width) / scale, height: CGFloat(uprightImg.height) / scale, scale: scale)
            
            renderedPages.append(RenderTask(
                logicalIndex: currentLogicalIndex,
                originalIndex: i,
                image: uprightImg,
                geometry: uprightGeom,
                pdfPage: wasRotated ? nil : cgPage
            ))
            currentLogicalIndex += 1
        } else {
            let splitParts = try await renderPagesSplitting(page, scale: scale, originalIndex: i, autocrop: config.autocrop, languages: config.languages, color: config.color)
            for part in splitParts {
                renderedPages.append(RenderTask(
                    logicalIndex: currentLogicalIndex,
                    originalIndex: i,
                    image: part.image,
                    geometry: part.geometry,
                    pdfPage: nil
                ))
                currentLogicalIndex += 1
            }
        }
    }

    let splitInfo = config.noSplit ? "nosplit" : "split rozkładówek"
    print("✂️  Tryb: \(splitInfo) gotowy. Liczba zadań: \(renderedPages.count)")

    let maxConcurrent = max(2, ProcessInfo.processInfo.activeProcessorCount - 1)
    var resultsDict: [Int: PageResult] = [:]

    try await withThrowingTaskGroup(of: PageResult.self) { group in
        var iterator = renderedPages.makeIterator()
        var running = 0

        func startNext() {
            guard let item = iterator.next() else { return }
            running += 1
            group.addTask {
                let result = try await processPageOCR(
                    image: item.image,
                    geometry: item.geometry,
                    logicalPageIndex: item.logicalIndex,
                    originalPageIndex: item.originalIndex,
                    languages: config.languages,
                    customWords: config.customWords,
                    pdfPage: item.pdfPage
                )

                let allLines = result.paragraphs.flatMap { $0.lines }
                let avgConfidence: Float = allLines.isEmpty
                    ? 0
                    : allLines.map { $0.confidence }.reduce(0, +) / Float(allLines.count)
                let footnoteCount = result.paragraphs.filter { $0.role == .footnote }.count
                let marginalCount = result.paragraphs.filter { $0.role == .marginal }.count

                print("   ✅ skan logiczny \(item.logicalIndex + 1) (org. \(item.originalIndex + 1)): " +
                      "\(result.paragraphs.count) akapitów (przypisy: \(footnoteCount), marginalia: \(marginalCount)), " +
                      "śr. pewność \(String(format: "%.2f", avgConfidence))")
                return result
            }
        }

        for _ in 0..<maxConcurrent { startNext() }

        while running > 0 {
            let result = try await group.next()!
            resultsDict[result.logicalPageIndex] = result
            running -= 1
            startNext()
        }
    }

    let orderedResults = renderedPages.map { resultsDict[$0.logicalIndex]! }

    // Zoptymalizowana, jednowątkowa pętla korekcji (NSSpellChecker)
    let finalResults: [PageResult]
    if config.fixDiacritics || config.splitMergedWords {
        var steps: [String] = []
        if config.fixDiacritics { steps.append("diakrytyki") }
        if config.splitMergedWords { steps.append("separacja zlepionych wyrazów") }
        if config.useContextualNLP { steps.append("zaawansowany model kontekstowy (BERT)") }
        print("🔤 Korekta tekstu (NSSpellChecker, pl): \(steps.joined(separator: " + "))...")

        let corrector = TextCorrectionEngine(useContextualNLP: config.useContextualNLP)
                if config.useContextualNLP {
                    await corrector.prepareEmbedding()
                }
                
                var correctedResults: [PageResult] = []
                
                // Wyodrębniamy bazowy kod języka z konfiguracji (np. "pl-PL" -> "pl")
                let preferredLanguage = config.languages.first?.split(separator: "-").first.map(String.init) ?? "pl"
                
                // Wykonujemy korektę sekwencyjnie
                for (idx, page) in orderedResults.enumerated() {
                    let correctedPage = corrector.correctSinglePage(
                        page,
                        fixDiacritics: config.fixDiacritics,
                        splitMergedWords: config.splitMergedWords,
                        customWords: config.customWords,
                        defaultLanguage: preferredLanguage // Przekazanie poprawnego języka
                    )
                    correctedResults.append(correctedPage)
                    
                    if (idx + 1) % 10 == 0 || (idx + 1) == orderedResults.count {
                        print("   ✍️ Skorygowano językowo: \(idx + 1)/\(orderedResults.count) stron...")
                    }
                }
        
        finalResults = correctedResults
        
        // --- ZAPIS INDEKSU NAZWISK (Wewnątrz prawidłowego zasięgu zmiennej 'corrector') ---
        let namesURL = outputURL.deletingPathExtension().appendingPathExtension("names.txt")
        let filteredNames = corrector.accumulatedNames.filter { $0.count > 2 }.sorted()
        let namesContent = filteredNames.joined(separator: "\n")
        
        try? namesContent.write(to: namesURL, atomically: true, encoding: String.Encoding.utf8)
        print("📝 Wyekstrahowano i zapisano \(filteredNames.count) unikalnych nazwisk do: \(namesURL.path)")
        
    } else {
        finalResults = orderedResults
    }

    // Zapis wyjściowego PDF z przekazaniem parametru lossless
    try writeOutputPDF(results: finalResults, to: outputURL, metadata: pdfMetadata, lossless: config.lossless)

    if let sidecarPath = config.sidecarPath {
        try writeSidecarText(results: finalResults, to: URL(fileURLWithPath: sidecarPath))
        print("📝 sidecar: \(sidecarPath)")
    }

    // Walidacja wygenerowanego pliku PDF pod kątem poprawności warstwy tekstowej i przeszukiwalności
    PDFValidator.verifyPDF(at: outputURL, expectedPageCount: finalResults.count)

    let attrs = try? FileManager.default.attributesOfItem(atPath: outputURL.path)
    var sizeStr = "?"
    if let sizeNum = attrs?[.size] as? Int64 {
        sizeStr = String(format: "%.1f MB", Double(sizeNum) / 1_048_576.0)
    }
    print("\n✅ Gotowe: \(outputURL.path) (\(sizeStr))")
}

// URUCHOMIENIE PROGRAMU
do {
    try await runPipeline()
} catch {
    FileHandle.standardError.write("❌ Błąd krytyczny: \(error)\n".data(using: .utf8)!)
    exit(1)
}


