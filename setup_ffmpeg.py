import os
import sys
import re
from pathlib import Path
import winreg

def check_ffmpeg_path():
    path_list = os.environ['Path'].split(';')
    ffmpeg_found = False

    for path in path_list:
        if 'ffmpeg' in path.lower() and 'bin' in path.lower():
            ffmpeg_found = True
            print("FFmpeg already installed.")
            break

    return ffmpeg_found

def add_ffmpeg_path_to_user_variable():
    ffmpeg_bin_path = Path('.\\ffmpeg\\bin')
    if ffmpeg_bin_path.is_dir():
        abs_path = str(ffmpeg_bin_path.resolve())
        
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_READ | winreg.KEY_WRITE
            )
            
            try:
                current_path, _ = winreg.QueryValueEx(key, "Path")
                if abs_path not in current_path:
                    new_path = f"{current_path};{abs_path}"
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                    print(f"Added FFmpeg path to user variable 'Path': {abs_path}")
                else:
                    print("FFmpeg path already exists in the user variable 'Path'.")
            finally:
                winreg.CloseKey(key)
        except WindowsError:
            print("Error: Unable to modify user variable 'Path'.")
            sys.exit(1)

    else:
        print("Error: ffmpeg\\bin folder not found in the current path.")
        sys.exit(1)

def main():
    if not check_ffmpeg_path():
       add_ffmpeg_path_to_user_variable()
            
if __name__ == "__main__":
    main()