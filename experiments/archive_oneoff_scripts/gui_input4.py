"""Attach to CTA thread, force IME English, type into field."""
import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32
imm32 = ctypes.windll.imm32
kernel32 = ctypes.windll.kernel32

# Get CTA window
hwnd = user32.GetForegroundWindow()
print(f"CTA hwnd: {hwnd}")

# Get CTA's thread ID
cta_tid = user32.GetWindowThreadProcessId(hwnd, None)
my_tid = kernel32.GetCurrentThreadId()
print(f"CTA thread: {cta_tid}, My thread: {my_tid}")

# Attach our input to CTA's thread
attached = user32.AttachThreadInput(my_tid, cta_tid, True)
print(f"AttachThreadInput: {attached}")
time.sleep(0.2)

# Now try to get IMM context from CTA's focused child
focus_hwnd = user32.GetFocus()
print(f"Focused child: {focus_hwnd}")

# Click on the input field first
user32.SetCursorPos(300, 78)
time.sleep(0.1)
user32.mouse_event(0x0002, 0, 0, 0, 0)
user32.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(0.5)

focus_hwnd2 = user32.GetFocus()
print(f"Focused child after click: {focus_hwnd2}")

# Get IMM context from the focused widget
himc = imm32.ImmGetContext(focus_hwnd2 if focus_hwnd2 else hwnd)
print(f"IMM context: {himc}")

if himc:
    conv = ctypes.wintypes.DWORD()
    sent = ctypes.wintypes.DWORD()
    imm32.ImmGetConversionStatus(himc, ctypes.byref(conv), ctypes.byref(sent))
    print(f"IME mode: {conv.value} (0=eng, 1+=chn)")
    
    # Force English
    if conv.value & 1:
        imm32.ImmSetConversionStatus(himc, 0, sent.value)
        time.sleep(0.2)
        print("Switched to English")
    
    imm32.ImmReleaseContext(focus_hwnd2 if focus_hwnd2 else hwnd, himc)

# Now select all and retype
# Ctrl+A
user32.keybd_event(0x11, 0, 0, 0)
user32.keybd_event(0x41, 0, 0, 0)
time.sleep(0.03)
user32.keybd_event(0x41, 0, 2, 0)
user32.keybd_event(0x11, 0, 2, 0)
time.sleep(0.3)

# Type new text (replaces selection)
for ch in 'p2505.DCE':
    vk = user32.VkKeyScanW(ord(ch))
    vk_code = vk & 0xFF
    need_shift = (vk >> 8) & 1
    if need_shift:
        user32.keybd_event(0x10, 0, 0, 0)
    user32.keybd_event(vk_code, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(vk_code, 0, 2, 0)
    if need_shift:
        user32.keybd_event(0x10, 0, 2, 0)
    time.sleep(0.05)

time.sleep(0.3)

# Detach
user32.AttachThreadInput(my_tid, cta_tid, False)

print("Done")
