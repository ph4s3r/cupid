import re
import threading

def timed_input(prompt, timeout=15):
    def input_thread(prompt, result):
        result.append(input(prompt))

    result = []
    thread = threading.Thread(target=input_thread, args=(prompt, result))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("\nTimeout! Continuing without input.")
        thread.join()  # Ensure the thread finishes execution
        return ''
    else:
        return result[0]
    
def sanitize(name):
    # replace any character that is not alphanumeric, a space, a hyphen, or an underscore
    sanitized_name = re.sub(r'[^\w\s-]', '', name)
    # replace spaces with underscores or hyphens for better compatibility
    sanitized_name = sanitized_name.replace(' ', '')
    return sanitized_name