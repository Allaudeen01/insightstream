try:
    with open('test_output.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open('test_output.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    if "Upload Response Content:" in line:
        print("FOUND CONTENT LINE:")
        print(line[:500])
    if "Generate Viz Failed:" in line:
        print(line)
    if "Response:" in line:
        print(line[:2000]) # Print more to see traceback
    if "Backend Error:" in line:
        print(line)
    if "Traceback" in line:
        print(line)
    if "Error in generate_visualizations:" in line:
        print(line)
