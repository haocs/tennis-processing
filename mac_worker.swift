import AppKit
import Foundation

// Give macOS a named, stable responsible app for Local Network privacy.
// The portable Python executor remains the only task-processing implementation.
let application = NSApplication.shared
application.setActivationPolicy(.accessory)
let resource = Bundle.main.url(forResource: "worker-launch", withExtension: "json")!
let dictionary = try JSONSerialization.jsonObject(with: Data(contentsOf: resource)) as! [String: String]
let process = Process()
process.executableURL = URL(fileURLWithPath: dictionary["python"]!)
process.arguments = [dictionary["script"]!, "--config", dictionary["config"]!]
process.currentDirectoryURL = URL(fileURLWithPath: dictionary["directory"]!)
var environment = ProcessInfo.processInfo.environment
environment["PYTHONHOME"] = dictionary["python_home"]!
environment["PYTHONPATH"] = dictionary["site_packages"]!
environment.removeValue(forKey: "__PYVENV_LAUNCHER__")
process.environment = environment
process.standardOutput = FileHandle.standardOutput
process.standardError = FileHandle.standardError
process.terminationHandler = { _ in DispatchQueue.main.async { application.terminate(nil) } }
signal(SIGTERM, SIG_IGN)
let termination = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termination.setEventHandler {
    if process.isRunning { process.terminate() }
    application.terminate(nil)
}
termination.resume()
try process.run()
application.run()
if process.isRunning { process.terminate(); process.waitUntilExit() }
