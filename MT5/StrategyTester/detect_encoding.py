import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read the first 10KB for detection
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    print(f"Detected encoding: {encoding} with confidence {confidence}")
    return encoding

if __name__ == "__main__":
    html_file_path = r"C:\Users\StdUser\Desktop\MyProjects\Backtesting\logs\AMD_10008151_output_report.html"  # Replace with your file path
    encoding = detect_encoding(html_file_path)
