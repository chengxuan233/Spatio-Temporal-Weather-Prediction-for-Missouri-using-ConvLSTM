import webbrowser
import time

INPUT_FILE = "subset_GLDAS_CLSM025_DA1_D_2.2_20250312_131341_.txt"

DELAY_BETWEEN_LINKS = 5

START_LINE = int(input("Enter the line number to start from (1-based index): "))


def open_links(start_line):
    with open(INPUT_FILE, "r") as f:
        all_urls = [line.strip() for line in f if line.strip().startswith("http")]

    if not all_urls:
        print("No valid URLs found in the input file!")
        return

    total_links = len(all_urls)

    # Validate start_line
    if start_line < 1 or start_line > total_links:
        print(f"Invalid start line! Choose between 1 and {total_links}.")
        return

    print(f"Resuming from line {start_line}/{total_links}...")

    for i, url in enumerate(all_urls[start_line - 1:], start=start_line):
        print(f"Opening link {i}/{total_links}: {url}")
        webbrowser.open(url)
        time.sleep(DELAY_BETWEEN_LINKS)

    print("All remaining links have been opened! Start downloads manually in the browser.")


# Run the script
if __name__ == "__main__":
    open_links(START_LINE)
