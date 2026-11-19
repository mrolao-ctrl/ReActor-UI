; ReActor Studio lightweight Windows installer.
; Packages the source code/launcher but not heavy Python runtimes or model weights.
; Models and Python dependencies are reused from the user's existing installation.
#define AppName "ReActor Studio"
#define TaskbarAppId "MRolao.ReActorStudio"
#define AppVersion "1.0.3"

[Setup]
AppId={{D979F4E0-079F-4230-9BD6-7CB41FE3B57A}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} Custom
DefaultDirName={localappdata}\Programs\ReActor Studio
DefaultGroupName=ReActor Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
OutputDir=..\dist
OutputBaseFilename=ReActor-Studio-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\reactor-studio.ico
WizardImageStretch=yes
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\assets\reactor-studio.ico
DisableWelcomePage=no
CloseApplications=yes

[Files]
Source: "..\run.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\modules\*"; DestDir: "{app}\modules"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\reactor-studio.png"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "..\assets\reactor-studio.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "Start-ReActor.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\ReActor Studio"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ""{app}\installer\Start-ReActor.ps1"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\reactor-studio.ico"; AppUserModelID: "{#TaskbarAppId}"
Name: "{autodesktop}\ReActor Studio"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ""{app}\installer\Start-ReActor.ps1"""; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\assets\reactor-studio.ico"; AppUserModelID: "{#TaskbarAppId}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ""{app}\installer\Start-ReActor.ps1"""; Description: "Launch ReActor Studio"; Flags: nowait postinstall skipifsilent
