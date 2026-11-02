; Inno Setup script for the Windows installer.
; Built by scripts/build_desktop.py, which passes AppVersion (from VERSION),
; SourceDir (the PyInstaller output folder) and OutputDir.

#ifndef AppVersion
  #error AppVersion must be defined (/DAppVersion=...)
#endif
#ifndef SourceDir
  #error SourceDir must be defined (/DSourceDir=...)
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

[Setup]
; Keep this AppId fixed: it is how upgrades find the existing install.
AppId={{7FC11677-0A42-427F-A57C-60FED2B6B1FD}
AppName=RadWright Screw
AppVersion={#AppVersion}
AppVerName=RadWright Screw {#AppVersion}
AppPublisher=Sang-Min Park
AppPublisherURL=https://github.com/grotyx/pedicle-screw-simulator
; Per-user install under %LOCALAPPDATA%\Programs: no UAC prompt.
PrivilegesRequired=lowest
DefaultDirName={autopf}\RadWright Screw
DefaultGroupName=RadWright Screw
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\PedicleScrewSimulator.exe
LicenseFile={#SourceDir}\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename=RadWright-Screw-{#AppVersion}-Windows-x64-Setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; The CUDA runtime makes the payload several GB; split it into Setup.exe + .bin
; slices instead of hitting the single-file size limit.
DiskSpanning=yes
DiskSliceSize=max
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
SetupIconFile=icons\radwright-screw.ico
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Replace the whole bundle on upgrade so files dropped by a newer build do not linger.
Type: filesandordirs; Name: "{app}\_internal"

[UninstallDelete]
; Updates can add files the installer never recorded; remove them too.
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\components.json"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\RadWright Screw"; Filename: "{app}\PedicleScrewSimulator.exe"
Name: "{group}\{cm:UninstallProgram,RadWright Screw}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\RadWright Screw"; Filename: "{app}\PedicleScrewSimulator.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\PedicleScrewSimulator.exe"; Description: "{cm:LaunchProgram,RadWright Screw}"; Flags: nowait postinstall skipifsilent
