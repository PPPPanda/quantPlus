param(
    [string]$Symbol = "p2509.DCE",
    [string]$StartDate = "2025/3/1",
    [string]$EndDate = "2025/8/31",
    [string]$Slippage = "0",
    [string]$Capital = "1000000"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32Focus {
    [DllImport("user32.dll")] public static extern IntPtr FindWindow(string c, string n);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
}
"@

# Focus CTA window
$h = [Win32Focus]::FindWindow([NullString]::Value, "CTA回测")
if ($h -ne [IntPtr]::Zero) {
    [Win32Focus]::SetForegroundWindow($h)
    Write-Host "Focused CTA backtest window"
    Start-Sleep -Milliseconds 500
} else {
    Write-Host "CTA window not found!"
    exit 1
}

function SetField($x, $y, $value) {
    cd E:\clawdbot_bridge\clawdbot_workspace\skills\windows-control
    node scripts/cli.js click --x $x --y $y 2>$null | Out-Null
    Start-Sleep -Milliseconds 400
    [System.Windows.Forms.SendKeys]::SendWait("{HOME}")
    Start-Sleep -Milliseconds 100
    [System.Windows.Forms.SendKeys]::SendWait("+{END}")
    Start-Sleep -Milliseconds 100
    [System.Windows.Forms.SendKeys]::SendWait("$value")
    Start-Sleep -Milliseconds 300
    Write-Host "Set field at ($x,$y) = $value"
}

# Symbol field (y≈100)
SetField 200 100 $Symbol

# Slippage field (y≈291)
SetField 200 291 $Slippage

# Capital field (y≈405)
SetField 200 405 $Capital

Write-Host "Done setting fields."
Write-Host "Note: Date fields (QDateEdit) may need manual adjustment."
