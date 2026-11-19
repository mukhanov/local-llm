// macos/widget.swift — native desktop monitor panel for local-llm.
//
// Built by `local-llm widget` into ~/Applications/MLX Stack.app.
// The app owns its Python snapshot API child and needs no terminal.
//
// Reproduces the curses monitor (local_llm/monitor.py) as a borderless
// desktop panel: it polls GET http://127.0.0.1:<port>/snapshot once a
// second, keeps the same 240-sample histories and decode-only tok/s
// logic, and renders the same sections — header chips, TOK/S main /
// TOK/S small / CONTEXT graphs, CPU + cores, clients / processes,
// MEM/SWP bars, top mem, logs (errors red). macOS 14+.

import AppKit
import Foundation
import SwiftUI

// MARK: - entry

@main
@MainActor
struct MLXStackApp {
    private static var retainedDelegate: AppDelegate?

    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        retainedDelegate = delegate   // NSApplication.delegate is non-retaining
        app.delegate = delegate
        app.setActivationPolicy(.accessory)   // no Dock icon, no menu bar
        app.appearance = NSAppearance(named: .darkAqua)
        app.run()
    }
}

/// "--port <n>" (or "--port=<n>"); default WIDGET_PORT of the launcher.
enum PortArg {
    static let value: Int = {
        let args = CommandLine.arguments
        var i = 0
        while i < args.count {
            if args[i] == "--port", i + 1 < args.count, let p = Int(args[i + 1]) {
                return p
            }
            if args[i].hasPrefix("--port="), let p = Int(args[i].dropFirst("--port=".count)) {
                return p
            }
            i += 1
        }
        return 19798
    }()
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let model = PanelModel(port: PortArg.value)
    var window: NSWindow?
    private var statusItem: NSStatusItem?
    private var apiProcess: Process?

    func applicationWillFinishLaunching(_ notification: Notification) {
        // accessory apps get no main menu; ⌘Q needs one
        let menu = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu(title: "MLXStack")
        appMenu.addItem(withTitle: "Quit MLXStack",
                        action: #selector(NSApplication.terminate(_:)),
                        keyEquivalent: "q")
        appItem.submenu = appMenu
        menu.addItem(appItem)
        NSApplication.shared.mainMenu = menu
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let view = ResizeGuardView()
        let host = NSHostingView(rootView: PanelView(model: model))
        host.autoresizingMask = [.width, .height]
        view.addSubview(host)

        let w = PanelWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1100, height: 800),
            styleMask: [.borderless, .resizable], backing: .buffered, defer: false)
        w.title = "MLX Stack"
        w.sharingType = .readOnly
        w.isOpaque = false
        w.backgroundColor = .clear
        w.hasShadow = true
        w.hidesOnDeactivate = false
        w.isMovableByWindowBackground = true
        w.level = .normal
        // Ordinary non-modal window: other apps can cover and activate freely.
        // Joins every (non-fullscreen) Space without floating above them.
        w.collectionBehavior = [.canJoinAllSpaces, .ignoresCycle]
        w.minSize = NSSize(width: 900, height: 700)
        w.contentMinSize = NSSize(width: 900, height: 700)
        w.contentView = view
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window = w

        startAPI()
        model.start()
        installStatusItem()
    }

    private func startAPI() {
        let python = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ollmlx/venv/bin/python")
        guard FileManager.default.fileExists(atPath: python.path) else {
            let alert = NSAlert()
            alert.messageText = "local-llm is not initialized"
            alert.informativeText = "Run local-llm once, then reopen MLX Stack."
            alert.alertStyle = .warning
            alert.runModal()
            return
        }
        let process = Process()
        process.executableURL = python
        process.arguments = ["-m", "local_llm.widget"]
        var env = ProcessInfo.processInfo.environment
        env["WIDGET_PORT"] = String(model.port)
        process.environment = env
        do {
            try process.run()
            apiProcess = process
        } catch {
            let alert = NSAlert()
            alert.messageText = "Cannot start the snapshot API"
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let apiProcess, apiProcess.isRunning {
            apiProcess.terminate()
        }
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        guard let button = item.button else { return }
        let image = NSImage(systemSymbolName: "terminal",
                            accessibilityDescription: "MLX Stack")
        image?.isTemplate = true
        button.image = image
        button.toolTip = "MLX Stack — monitor and models"
        button.setAccessibilityTitle("MLX Stack")
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        statusItem = item
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        window?.makeKeyAndOrderFront(nil)
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
            self.window?.makeKeyAndOrderFront(nil)
        }
        menu.removeAllItems()
        let open = NSMenuItem(
            title: window?.isVisible == true ? "Hide monitor" : "Open monitor",
            action: #selector(togglePanel), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        menu.addItem(.separator())

        let title = model.canSwitch ? "Load model" : "Models — terminal stack active"
        let header = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        for name in model.availableModels {
            let item = NSMenuItem(title: name, action: #selector(selectModel(_:)),
                                  keyEquivalent: "")
            item.target = self
            item.representedObject = name
            item.state = model.derived?.model == name ? .on : .off
            item.isEnabled = model.canSwitch && !model.modelActionInProgress
            menu.addItem(item)
        }
        if model.availableModels.isEmpty {
            let empty = NSMenuItem(title: "No downloaded models", action: nil,
                                   keyEquivalent: "")
            empty.isEnabled = false
            menu.addItem(empty)
        }
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Exit", action: #selector(quit),
                              keyEquivalent: "")
        quit.target = self
        menu.addItem(quit)
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        model.runModel(name)
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    @objc private func togglePanel() {
        guard let window else { return }
        if window.isVisible {
            window.orderOut(nil)
        } else {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false   // the status item owns the app lifetime; Cmd+Q terminates
    }
}

// MARK: - borderless window with real edge resizing

final class PanelWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

/// Content-view wrapper that owns an 7pt resize rim around the hosted
/// SwiftUI panel: borderless windows get no AppKit resize tracking, and
/// interior clicks still fall through to the window background drag.
final class ResizeGuardView: NSView {
    private let rim: CGFloat = 7

    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { false }

    private func zone(_ p: NSPoint) -> (left: Bool, right: Bool, bottom: Bool, top: Bool) {
        let b = bounds
        return (p.x < rim, p.x > b.maxX - rim, p.y < rim, p.y > b.maxY - rim)
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        let z = zone(point)
        if z.left || z.right || z.bottom || z.top { return self }
        if point.y > bounds.height - 55 { return self }  // header drag strip
        return super.hitTest(point)
    }

    override func resetCursorRects() {
        let b = bounds
        addCursorRect(NSRect(x: 0, y: 0, width: b.width, height: rim), cursor: .resizeUpDown)
        addCursorRect(NSRect(x: 0, y: b.height - rim, width: b.width, height: rim),
                      cursor: .resizeUpDown)
        addCursorRect(NSRect(x: 0, y: 0, width: rim, height: b.height), cursor: .resizeLeftRight)
        addCursorRect(NSRect(x: b.width - rim, y: 0, width: rim, height: b.height),
                      cursor: .resizeLeftRight)
        addCursorRect(NSRect(x: rim, y: b.height - 55, width: b.width - 2 * rim,
                             height: 55 - rim), cursor: .openHand)
    }

    override func mouseDown(with event: NSEvent) {
        guard let window = window else { return }
        let point = convert(event.locationInWindow, from: nil)
        let z = zone(point)
        guard z.left || z.right || z.bottom || z.top else {
            if point.y > bounds.height - 55 {
                let startMouse = NSEvent.mouseLocation
                let origin = window.frame.origin
                while let ev = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp]) {
                    if ev.type == .leftMouseUp { break }
                    let current = NSEvent.mouseLocation
                    window.setFrameOrigin(NSPoint(
                        x: origin.x + current.x - startMouse.x,
                        y: origin.y + current.y - startMouse.y))
                }
            } else {
                super.mouseDown(with: event)
            }
            return
        }
        let startMouse = NSEvent.mouseLocation
        let f0 = window.frame
        let minW = window.contentMinSize.width
        let minH = window.contentMinSize.height
        while true {
            guard let ev = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp]) else {
                break
            }
            if ev.type == .leftMouseUp { break }
            let m = NSEvent.mouseLocation
            let dx = m.x - startMouse.x
            let dy = m.y - startMouse.y
            var width = f0.width
            var height = f0.height
            var x = f0.minX
            var y = f0.minY
            if z.right { width = max(minW, f0.width + dx) }
            if z.top { height = max(minH, f0.height + dy) }
            if z.left { width = max(minW, f0.width - dx); x = f0.maxX - width }
            if z.bottom { height = max(minH, f0.height - dy); y = f0.maxY - height }
            window.setFrame(NSRect(x: x, y: y, width: width, height: height), display: true)
        }
    }

    override func layout() {
        super.layout()
        if let host = subviews.first {
            host.frame = bounds
        }
    }
}

/// Native wheel/trackpad events for the terminal-style log scrollback.
struct ScrollWheelTarget: NSViewRepresentable {
    let onScroll: (Double) -> Void

    func makeNSView(context: Context) -> WheelTargetView {
        let view = WheelTargetView()
        view.onScroll = onScroll
        return view
    }

    func updateNSView(_ view: WheelTargetView, context: Context) {
        view.onScroll = onScroll
    }
}

final class WheelTargetView: NSView {
    var onScroll: (Double) -> Void = { _ in }

    override func scrollWheel(with event: NSEvent) {
        onScroll(event.scrollingDeltaY)
    }
}

/// Dark, slightly translucent desktop material behind the panel.
struct VisualEffectBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .underWindowBackground
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}

// MARK: - snapshot payload

struct Snapshot: Decodable {
    struct Port: Decodable {
        let label: String
        let port: Int
        let up: Bool
    }

    struct Ports: Decodable {
        let main: Port
        let small: Port?
        let litellm: Port
    }

    struct CPU: Decodable {
        let total: Double
        let cores: [Double]
    }

    struct Mem: Decodable {
        let used: Int64
        let total: Int64
        let available: Int64
        let percent: Double
    }

    struct Swap: Decodable {
        let used: Int64
        let total: Int64
        let percent: Double
    }

    struct Proc: Decodable {
        let name: String
        let pid: Int
        let cpu: Double
        let rss: Int64
        let rssPercent: Double
        let threads: Int
    }

    struct ProcSummary: Decodable {
        let ours: Int64
        let others: Int64
        let oursPercent: Double
    }

    struct Eater: Decodable {
        let name: String
        let pid: Int
        let rss: Int64
    }

    struct Perf: Decodable {
        let liveSeq: Int
        let liveRate: Double
        let preSeq: Int
        let preRate: Double
        let summary: String
        let context: Int?
        let active: Int?
    }

    struct Client: Decodable {
        let name: String
        let command: String
    }

    struct LogLine: Decodable {
        let tag: String
        let text: String
        let error: Bool
    }

    let model: String
    let backend: String
    let uptimeSeconds: Int
    let maxInput: Int
    let availableModels: [String]
    let canSwitch: Bool
    let ports: Ports
    let cpu: CPU
    let memory: Mem
    let swap: Swap?
    let processes: [Proc]
    let processSummary: ProcSummary
    let topEaters: [Eater]
    let main: Perf
    let small: Perf?
    let clients: [Client]
    let logs: [LogLine]
}

// MARK: - local histories (the curses deques, 240 samples)

struct Ring {
    private(set) var values: [Double] = []
    private let cap: Int

    init(capacity: Int = 240) {
        self.cap = capacity
    }

    mutating func append(_ v: Double) {
        values.append(v)
        if values.count > cap {
            values.removeFirst(values.count - cap)
        }
    }

    var maxV: Double { values.max() ?? 0 }
}

/// Per-model decode state, mirroring monitor.py's `tok` dicts.
struct TokSt {
    var liveSeq = -1
    var preSeq = -1
    var idle = 1 << 30
    var cur = 0.0
    var txt = "idle"
    var summary = ""
    var peak = 0.0
}

// MARK: - immutable view model built once per poll

struct Chip {
    let text: String
    let up: Bool
}

struct Row {
    let text: String
    let color: Color
}

struct TokCol {
    let title: String                 // "TOK/S main"
    let barFrac: Double
    let barSuffix: String             // "]  12.3 tok/s"
    let boxTitle: String              // " tok/s 0–75 · avg 80 · max 120 "
    let samples: [Double]
    let scale: Double
    let footer: String                // "last: …" / placeholder
}

struct CtxCol {
    let barFrac: Double
    let barSuffix: String             // "] 4.1K/110K"
    let samples: [Double]
    let scale: Double                 // max_input (or observed max)
    let footer: String                // "TTFT … · uncached … · active …"
}

struct Derived {
    let model: String
    let uptime: String
    let chips: [Chip]
    let coreCount: Int
    let tokMain: TokCol
    let tokSmall: TokCol?
    let ctx: CtxCol
    let cpuTotal: Double
    let cpuSuffix: String
    let cpuHist: [Double]
    let cores: [Double]
    let clients: [(name: String, command: String)]
    let procRows: [Row]
    let memFrac: Double
    let memSuffix: String
    let memLow: Bool
    let swapFrac: Double?
    let swapSuffix: String?
    let topMem: Row?
    let logRows: [Row]
}

// MARK: - poller + state machine

@MainActor
final class PanelModel: ObservableObject {
    let port: Int

    @Published private(set) var derived: Derived?
    @Published private(set) var offline = true
    @Published private(set) var everConnected = false
    @Published private(set) var availableModels: [String] = []
    @Published private(set) var canSwitch = false
    @Published private(set) var mainRunning = false
    @Published private(set) var modelActionInProgress = false
    @Published var logOffset = 0
    var scrollRemainder = 0.0

    private var tokMain = TokSt()
    private var tokSmall = TokSt()
    private var ringTok = Ring()
    private var ringTokS = Ring()
    private var ringCtx = Ring()
    private var ringCpu = Ring()
    private var ctxCur = 0

    private let session: URLSession
    private var pollTask: Task<Void, Never>?

    private static let doneRx = try! NSRegularExpression(
        pattern: ##"^(\d+) tok.*? prompt (\d+) tok, (\d+) cached(?: \(([\d.]+)s to first\))?"##)

    init(port: Int) {
        self.port = port
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 2
        cfg.timeoutIntervalForResource = 3
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        session = URLSession(configuration: cfg)
    }

    func start() {
        pollTask?.cancel()
        guard let url = URL(string: "http://127.0.0.1:\(port)/snapshot") else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll(url: url)
                try? await Task.sleep(for: .seconds(1))   // sequential: never overlaps
            }
        }
    }

    func runModel(_ name: String) {
        guard canSwitch, !modelActionInProgress,
              let url = URL(string: "http://127.0.0.1:\(port)/run") else { return }
        modelActionInProgress = true
        Task {
            defer { modelActionInProgress = false }
            do {
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.timeoutInterval = 40
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: ["model": name])
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                if status != 202 {
                    let payload = try? JSONSerialization.jsonObject(with: data) as? [String: String]
                    throw NSError(domain: "MLXStack", code: status,
                                  userInfo: [NSLocalizedDescriptionKey:
                                      payload?["error"] ?? "model launch failed"])
                }
            } catch {
                let alert = NSAlert()
                alert.messageText = "Cannot load model"
                alert.informativeText = error.localizedDescription
                alert.alertStyle = .warning
                alert.runModal()
            }
        }
    }

    private func poll(url: URL) async {
        do {
            let (data, _) = try await session.data(from: url)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let snap = try decoder.decode(Snapshot.self, from: data)
            apply(snap)
            everConnected = true
            offline = false
        } catch {
            offline = true   // histories are kept; the view shows the offline state
        }
    }

    /// Same decode bookkeeping as monitor.py's draw_toks(): a changed
    /// live_seq shows live_rate (and feeds the graph), a changed pre_seq
    /// shows prefill, anything else decays to idle after 3 grace ticks.
    private func stepTok(_ st: inout TokSt, _ p: Snapshot.Perf, _ ring: inout Ring) {
        let liveNew = p.liveSeq != st.liveSeq
        let preNew = p.preSeq != st.preSeq
        if liveNew {
            st.idle = 0
            st.cur = p.liveRate
            st.txt = String(format: "%5.1f tok/s", p.liveRate)
            st.peak = max(st.peak, p.liveRate)
        } else if preNew {
            st.idle = 0
            st.cur = p.preRate
            st.txt = String(format: "%4.0f tok/s prefill", p.preRate)
        } else {
            st.idle += 1
            if st.idle > 3 {
                st.cur = 0.0
                st.txt = "idle"
            }
        }
        st.liveSeq = p.liveSeq
        st.preSeq = p.preSeq
        st.summary = p.summary
        // decode-only graph: zeros mark idle/prefill periods
        ring.append(liveNew ? p.liveRate : 0.0)
    }

    private func stepIdle(_ st: inout TokSt, _ ring: inout Ring) {
        st.idle += 1
        if st.idle > 3 {
            st.cur = 0.0
            st.txt = "idle"
        }
        ring.append(0.0)
    }

    private func tokCol(_ title: String, _ st: TokSt, _ ring: Ring) -> TokCol {
        var scale = 25.0
        let tmax = ring.maxV
        if tmax > 0 {
            scale = max(25.0, Double((Int(tmax) + 24) / 25 * 25))   // ceil to a 25 step
        }
        let active = ring.values.filter { $0 > 0 }
        let avg = active.isEmpty
            ? "–"
            : String(format: "%.0f", active.reduce(0, +) / Double(active.count))
        var boxTitle = " tok/s 0–\(Int(scale)) "
        if st.peak > 0 {
            boxTitle = " tok/s 0–\(Int(scale)) · avg \(avg) · max \(String(format: "%.0f", st.peak)) "
        }
        let footer = st.summary.isEmpty
            ? "no completions yet — tok/s appears during generation"
            : "last: " + st.summary
        return TokCol(title: title,
                      barFrac: scale > 0 ? min(1, max(0, st.cur / scale)) : 0,
                      barSuffix: "] \(st.txt)",
                      boxTitle: boxTitle,
                      samples: ring.values,
                      scale: scale,
                      footer: footer)
    }

    private func apply(_ snap: Snapshot) {
        availableModels = snap.availableModels
        canSwitch = snap.canSwitch
        mainRunning = snap.ports.main.up
        stepTok(&tokMain, snap.main, &ringTok)
        if let sp = snap.small {
            stepTok(&tokSmall, sp, &ringTokS)
        } else if snap.ports.small != nil {
            stepIdle(&tokSmall, &ringTokS)
        }
        if let c = snap.main.context { ctxCur = c }
        ringCtx.append(Double(ctxCur))
        ringCpu.append(snap.cpu.total)

        let chips: [Chip] = [snap.ports.main, snap.ports.small, snap.ports.litellm]
            .compactMap { $0 }
            .map { Chip(text: "\($0.label) :\($0.port) \($0.up ? "●up" : "○down")", up: $0.up) }

        // context: latest non-nil context against the hard cap
        let total = snap.maxInput > 0 ? Double(snap.maxInput) : Double(max(ctxCur, 1))
        var ttft = "–"
        var missed = "–"
        if let m = Self.doneRx.firstMatch(in: snap.main.summary,
                                          range: NSRange(snap.main.summary.startIndex...,
                                                         in: snap.main.summary)) {
            func group(_ i: Int) -> String? {
                guard let r = Range(m.range(at: i), in: snap.main.summary) else { return nil }
                return String(snap.main.summary[r])
            }
            if let ps = group(2), let cs = group(3), let pi = Int(ps), let ci = Int(cs) {
                missed = tokens(pi - ci)
            }
            if let ts = group(4) {
                ttft = String(format: "%.1fs", Double(ts) ?? 0)
            }
        }
        let activeTxt = snap.main.active.map(String.init) ?? "–"
        let ctx = CtxCol(barFrac: min(1, Double(ctxCur) / total),
                         barSuffix: "] \(tokens(ctxCur))/\(tokens(Int(total)))",
                         samples: ringCtx.values,
                         scale: total,
                         footer: "TTFT \(ttft) · uncached \(missed) · active \(activeTxt)")

        // process table + RAM ledger
        var procRows: [Row] = snap.processes.map { p in
            Row(text: rpad(p.name, 14) + " pid " + rpad(String(p.pid), 7)
                    + String(format: " %5.1f%% cpu", p.cpu)
                    + "  " + lpad(humanBytes(p.rss), 7) + " rss"
                    + String(format: " %4.1f%% ram", p.rssPercent)
                    + "  \(p.threads) thr",
                 color: heat(min(100, p.cpu)))
        }
        if !procRows.isEmpty {
            procRows.append(Row(text: "llm \(humanBytes(snap.processSummary.ours))"
                    + String(format: " · %.0f%% of RAM", snap.processSummary.oursPercent)
                    + " · others \(humanBytes(snap.processSummary.others))"
                    + " · available \(humanBytes(snap.memory.available))",
                 color: heat(snap.memory.percent)))
        } else {
            procRows.append(Row(text: "no mlx/litellm processes found", color: C.red))
        }

        // memory / swap
        let low = snap.memory.available < 10 * 1_073_741_824   // LOW_AVAILABLE_GB
        let memSuffix = low
            ? String(format: "] %3.0f%%  ⚠ OOM risk: %@ free",
                     snap.memory.percent, humanBytes(snap.memory.available))
            : String(format: "] %3.0f%%  %@/%@",
                     snap.memory.percent, humanBytes(snap.memory.used),
                     humanBytes(snap.memory.total))
        var swapFrac: Double?
        var swapSuffix: String?
        if let sw = snap.swap, sw.total > 0 {
            swapFrac = sw.percent / 100
            swapSuffix = String(format: "] %3.0f%%  %@/%@", sw.percent,
                                humanBytes(sw.used), humanBytes(sw.total))
        }

        // heaviest third-party apps — the kill candidates
        let topMem: Row? = snap.topEaters.isEmpty
            ? nil
            : Row(text: " top mem (kill <pid>): "
                    + snap.topEaters.map { "\($0.name) \(humanBytes($0.rss))(\($0.pid))" }
                        .joined(separator: " · "),
                  color: low ? C.red : C.dim)

        // errors red, then the dim tail
        var logRows = snap.logs.map {
            Row(text: "[\($0.tag)] \($0.text)", color: $0.error ? C.red : C.dim)
        }
        if logRows.isEmpty {
            logRows = [Row(text: "no errors · logs empty", color: C.green)]
        }

        derived = Derived(
            model: snap.model,
            uptime: fmtUptime(snap.uptimeSeconds),
            chips: chips,
            coreCount: snap.cpu.cores.count,
            tokMain: tokCol("TOK/S main", tokMain, ringTok),
            tokSmall: snap.ports.small != nil
                ? tokCol("TOK/S small", tokSmall, ringTokS)
                : nil,
            ctx: ctx,
            cpuTotal: snap.cpu.total,
            cpuSuffix: String(format: "] %5.1f%%", snap.cpu.total),
            cpuHist: ringCpu.values,
            cores: snap.cpu.cores,
            clients: snap.clients.map { ($0.name, $0.command) },
            procRows: procRows,
            memFrac: snap.memory.percent / 100,
            memSuffix: memSuffix,
            memLow: low,
            swapFrac: swapFrac,
            swapSuffix: swapSuffix,
            topMem: topMem,
            logRows: logRows)
    }
}

// MARK: - formatting (same shapes as ui.py / monitor.py)

enum C {
    static let green = Color(red: 0.30, green: 0.85, blue: 0.45)
    static let yellow = Color(red: 0.95, green: 0.78, blue: 0.30)
    static let red = Color(red: 0.96, green: 0.38, blue: 0.34)
    static let cyan = Color(red: 0.38, green: 0.85, blue: 0.95)
    static let fg = Color(white: 0.86)
    static let dim = Color(white: 0.52)
    static let barBg = Color(white: 0.24)
}

func heat(_ pct: Double) -> Color {   // green -> yellow -> red
    pct < 60 ? C.green : pct < 85 ? C.yellow : C.red
}

func mono(_ size: CGFloat = 12, _ weight: Font.Weight = .regular) -> Font {
    .system(size: size, weight: weight, design: .monospaced)
}

/// ui.human_bytes: B/K/M as integers, G with one decimal.
func humanBytes(_ n: Int64) -> String {
    var v = Double(n)
    for u in ["B", "K", "M"] {
        if v < 1024 {
            return String(format: "%.0f%@", v, u)
        }
        v /= 1024
    }
    return String(format: "%.1fG", v)
}

/// monitor.py's tokens(): 999 -> "999", 1500 -> "2K", 15000 -> "15.0K".
func tokens(_ n: Int) -> String {
    if n >= 10000 { return String(format: "%.1fK", Double(n) / 1000) }
    if n >= 1000 { return String(format: "%.0fK", Double(n) / 1000) }
    return String(n)
}

func fmtUptime(_ s: Int) -> String {
    let h = s / 3600, m = (s % 3600) / 60, sec = s % 60
    return h > 0 ? String(format: "%d:%02d:%02d", h, m, sec)
                 : String(format: "%d:%02d", m, sec)
}

func rpad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
}

func lpad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : String(repeating: " ", count: n - s.count) + s
}

// MARK: - panel view

struct PanelView: View {
    @ObservedObject var model: PanelModel

    var body: some View {
        ZStack {
            VisualEffectBackground()
            Color.black.opacity(0.55)
            if model.derived != nil, !model.offline {
                PanelContent(model: model)
                    .padding(14)
            } else {
                OfflineView(model: model)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .contextMenu {
            Button("Quit MLXStack") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q", modifiers: .command)
        }
    }
}

struct OfflineView: View {
    @ObservedObject var model: PanelModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(" MLX Stack")
                .font(mono(15, .bold))
                .foregroundStyle(C.cyan)
            Spacer()
            HStack {
                Spacer()
                VStack(spacing: 10) {
                    if model.everConnected {
                        Text("offline — cannot reach 127.0.0.1:\(model.port)")
                            .font(mono(14, .semibold))
                            .foregroundStyle(C.red)
                        Text("retrying every second · graphs resume on reconnect")
                            .font(mono(12))
                            .foregroundStyle(C.dim)
                    } else {
                        Text("waiting for the snapshot API on 127.0.0.1:\(model.port)")
                            .font(mono(14, .semibold))
                            .foregroundStyle(C.fg)
                        Text("polling once per second…")
                            .font(mono(12))
                            .foregroundStyle(C.dim)
                    }
                }
                Spacer()
            }
            Spacer()
            Text(" ⌘Q — quit · right-click — menu · snapshot 127.0.0.1:\(model.port)/snapshot")
                .font(mono(11))
                .foregroundStyle(C.dim)
                .lineLimit(1)
        }
        .padding(16)
    }
}

struct PanelContent: View {
    @ObservedObject var model: PanelModel
    private var d: Derived { model.derived! }
    private var port: Int { model.port }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            VStack(alignment: .leading, spacing: 7) {
                Text(" MLX Stack — \(d.model)")
                    .font(mono(15, .bold))
                    .foregroundStyle(C.cyan)
                    .lineLimit(1)
                headerLine
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(alignment: .top, spacing: 10) {
                tokColumn(d.tokMain)
                if let s = d.tokSmall {
                    tokColumn(s)
                }
                ctxColumn(d.ctx)
            }

            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 6) {
                    TermBar(label: " CPU [", frac: d.cpuTotal / 100, suffix: d.cpuSuffix)
                    Box(" CPU history ") {
                        GraphView(samples: d.cpuHist, scale: 100, colored: { heat($0) })
                            .frame(height: 70)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                Box(" cores ") {
                    coresGrid
                }
            }

            HStack(alignment: .top, spacing: 10) {
                Box(" clients ") {
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(d.clients.enumerated()), id: \.offset) { _, c in
                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(c.command, forType: .string)
                            } label: {
                                Text(rpad(c.name, 6) + "  " + c.command)
                                    .font(mono(12, .medium))
                                    .foregroundStyle(C.fg)
                                    .lineLimit(1)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(c.command)
                            .help("Click to copy command")
                            .contextMenu {
                                Button("Copy command") {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(c.command, forType: .string)
                                }
                            }
                        }
                    }
                }
                Box(" processes ") {
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(d.procRows.enumerated()), id: \.offset) { _, r in
                            Text(r.text)
                                .font(mono(11.5))
                                .foregroundStyle(r.color)
                                .lineLimit(1)
                        }
                    }
                }
            }

            TermBar(label: " MEM [", frac: d.memFrac, suffix: d.memSuffix,
                    fill: d.memLow ? C.red : nil,
                    suffixColor: d.memLow ? C.red : C.fg)
            if d.swapFrac != nil, let swSuffix = d.swapSuffix {
                TermBar(label: " SWP [", frac: d.swapFrac ?? 0, suffix: swSuffix)
            }
            if let tm = d.topMem {
                Text(tm.text)
                    .font(mono(11.5))
                    .foregroundStyle(tm.color)
                    .lineLimit(1)
            }

            Box(model.logOffset > 0 ? " logs · \(model.logOffset) from live " : " logs ") {
                GeometryReader { g in
                    let visible = max(1, Int(g.size.height / 13))
                    let maxOffset = max(0, d.logRows.count - visible)
                    let offset = min(model.logOffset, maxOffset)
                    let end = d.logRows.count - offset
                    let start = max(0, end - visible)
                    VStack(alignment: .leading, spacing: 1) {
                        ForEach(Array(d.logRows[start..<end].enumerated()), id: \.offset) { _, r in
                            Text(r.text)
                                .font(mono(11))
                                .foregroundStyle(r.color)
                                .lineLimit(1)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity,
                           alignment: .bottomLeading)
                    .overlay(ScrollWheelTarget { delta in
                        model.scrollRemainder += delta
                        let steps = Int(model.scrollRemainder / 12)
                        if steps != 0 {
                            model.logOffset = max(0, min(maxOffset, model.logOffset + steps))
                            model.scrollRemainder -= Double(steps) * 12
                        }
                    })
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)

            Text(" ⌘Q — quit · scroll — logs · right-click — menu · snapshot 127.0.0.1:\(port)/snapshot")
                .font(mono(11))
                .foregroundStyle(C.dim)
                .lineLimit(1)
        }
    }

    private var headerLine: some View {
        var s = AttributedString("uptime \(d.uptime)   ")
        s.foregroundColor = C.fg
        for chip in d.chips {
            var t = AttributedString(chip.text + "   ")
            t.foregroundColor = chip.up ? C.green : C.red
            s.append(t)
        }
        var tail = AttributedString("cores \(d.coreCount)")
        tail.foregroundColor = C.fg
        s.append(tail)
        return Text(s)
            .font(mono(12))
            .lineLimit(1)
    }

    private func tokColumn(_ t: TokCol) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            TermBar(label: " \(t.title) [", frac: t.barFrac, suffix: t.barSuffix,
                    fill: C.green)
            Box(t.boxTitle) {
                GraphView(samples: t.samples, scale: t.scale, colored: { _ in C.green })
                    .frame(height: 96)
            }
            Text(t.footer)
                .font(mono(11))
                .foregroundStyle(C.dim)
                .lineLimit(1)
                .padding(.horizontal, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func ctxColumn(_ t: CtxCol) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            TermBar(label: " CONTEXT [", frac: t.barFrac, suffix: t.barSuffix)
            Box(" context history ") {
                GraphView(samples: t.samples, scale: t.scale,
                          colored: { heat(100 * $0 / max(t.scale, 1)) })
                    .frame(height: 96)
            }
            Text(t.footer)
                .font(mono(11))
                .foregroundStyle(C.dim)
                .lineLimit(1)
                .padding(.horizontal, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var coresGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8)],
                  alignment: .leading, spacing: 5) {
            ForEach(Array(d.cores.enumerated()), id: \.offset) { i, v in
                HStack(spacing: 6) {
                    Text(String(format: "C%02d", i))
                        .foregroundStyle(heat(v))
                    MiniBar(frac: v / 100)
                        .frame(width: 36, height: 9)
                    Text(String(format: "%3.0f%%", v))
                        .foregroundStyle(heat(v))
                }
                .font(mono(12))
            }
        }
    }
}

// MARK: - terminal primitives

/// label[■■■■□□□□] suffix — the put_bar() of the curses UI.
struct TermBar: View {
    let label: String
    let frac: Double
    let suffix: String
    var fill: Color? = nil       // nil = heat(frac * 100)
    var suffixColor: Color = C.fg

    var body: some View {
        HStack(spacing: 0) {
            Text(label)
                .font(mono(12, .semibold))
                .foregroundStyle(C.fg)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Rectangle().fill(C.barBg)
                    Rectangle()
                        .fill(fill ?? heat(frac * 100))
                        .frame(width: max(0, min(1, frac)) * g.size.width)
                }
            }
            .frame(height: 12)
            Text(suffix)
                .font(mono(12, .semibold))
                .foregroundStyle(suffixColor)
                .lineLimit(1)
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

struct MiniBar: View {
    let frac: Double

    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Rectangle().fill(C.barBg)
                Rectangle()
                    .fill(heat(frac * 100))
                    .frame(width: max(0, min(1, frac)) * g.size.width)
            }
        }
    }
}

/// Column sparkline: one bar per sample, newest on the right.
struct GraphView: View {
    let samples: [Double]
    let scale: Double
    let colored: (Double) -> Color

    var body: some View {
        Canvas { ctx, size in
            let n = samples.count
            guard n > 0, scale > 0 else { return }
            let cw = size.width / CGFloat(n)
            for (i, v) in samples.enumerated() {
                let f = CGFloat(max(0, min(1, v / scale)))
                guard f > 0 else { continue }
                let h = f * size.height
                let rect = CGRect(x: CGFloat(i) * cw, y: size.height - h,
                                  width: max(0.5, cw - 0.6), height: h)
                ctx.fill(Path(rect), with: .color(colored(v)))
            }
        }
    }
}

/// Bordered box with a title chip, like the curses ┌ logs ┐ frames.
struct Box<Content: View>: View {
    private let title: String
    private let makeContent: () -> Content

    init(_ title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.makeContent = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(mono(11))
                .foregroundStyle(C.dim)
                .padding(.horizontal, 6)
                .padding(.top, 5)
            makeContent()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.bottom, 7)
        .padding(.horizontal, 4)
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .strokeBorder(C.dim.opacity(0.5), lineWidth: 1))
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
    }
}
