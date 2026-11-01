param(
  [string]$Port = 'COM10',
  [int]$BootWaitMs = 2600
)
# Switch K230 to debug mode (REPL back to CanMV IDE) without unplugging.
# Chain: this script -> ESP32 USB 'gdebug' -> $E,DEBUG! -> K230 writes offline_off + reset
$ErrorActionPreference = 'Stop'
$p = New-Object System.IO.Ports.SerialPort $Port,115200,'None',8,'One'
$p.Encoding = [System.Text.Encoding]::UTF8
$p.ReadTimeout = 200
$p.Open()
try {
  $p.DtrEnable = $true
  $p.RtsEnable = $true
  Start-Sleep -Milliseconds $BootWaitMs
  $p.DiscardInBuffer()
  $p.WriteLine('gdebug')
  Start-Sleep -Milliseconds 900
  $out = $p.ReadExisting()
} finally {
  $p.Close()
}
if ($out -match [regex]::Escape('-> K230 $E,DEBUG!')) {
  Write-Host 'OK: ESP32 forwarded $E,DEBUG!; K230 is writing offline_off and resetting. Wait 2-3s, then connect CanMV IDE.'
  exit 0
}
Write-Host 'FAIL: ESP32 did not confirm $E,DEBUG!. Raw output:'
Write-Host $out
Write-Host 'Check: ESP32 firmware has gdebug, and the COM port is correct.'
exit 1
