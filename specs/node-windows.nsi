; DistribAI Node Windows Installer
!define PRODUCT_NAME "DistribAI Node"
!define PRODUCT_VERSION "0.8.0"
!define PRODUCT_PUBLISHER "DistribAI"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\DistribAI-Node.exe"

SetCompressor lzma
RequestExecutionLevel admin

!include "MUI2.nsh"

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "..\dist\DistribAI-Node-Windows-Setup.exe"
InstallDir "$PROGRAMFILES64\DistribAI Node"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""
ShowInstDetails show
ShowUnInstDetails show

Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite ifnewer
  File /r "..\dist\DistribAI-Node-Windows\*"
  CreateDirectory "$SMPROGRAMS\DistribAI Node"
  CreateShortcut "$SMPROGRAMS\DistribAI Node\DistribAI Node.lnk" "$INSTDIR\DistribAI-Node.exe"
  CreateShortcut "$DESKTOP\DistribAI Node.lnk" "$INSTDIR\DistribAI-Node.exe"
  ; Auto-start the node when the contributor logs in.
  CreateDirectory "$SMSTARTUP\DistribAI Node"
  CreateShortcut "$SMSTARTUP\DistribAI Node\DistribAI-Node.lnk" "$INSTDIR\DistribAI-Node.exe"
SectionEnd

Section -AdditionalIcons
  CreateShortcut "$SMPROGRAMS\DistribAI Node\Uninstall.lnk" "$INSTDIR\uninst.exe"
SectionEnd

Section -Post
  WriteUninstaller "$INSTDIR\uninst.exe"
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\DistribAI-Node.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DistribAI-Node" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DistribAI-Node" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DistribAI-Node" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DistribAI-Node" "Publisher" "${PRODUCT_PUBLISHER}"
SectionEnd

Section Uninstall
  Delete "$INSTDIR\uninst.exe"
  RMDir /r "$INSTDIR"
  Delete "$DESKTOP\DistribAI Node.lnk"
  Delete "$SMPROGRAMS\DistribAI Node\Uninstall.lnk"
  Delete "$SMPROGRAMS\DistribAI Node\DistribAI Node.lnk"
  RMDir "$SMPROGRAMS\DistribAI Node"
  Delete "$SMSTARTUP\DistribAI Node\DistribAI-Node.lnk"
  RMDir "$SMSTARTUP\DistribAI Node"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\DistribAI-Node"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
SectionEnd
