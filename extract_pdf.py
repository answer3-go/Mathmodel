import sys
import PyPDF2


def extract_first_pages(pdf_path: str, pages: int = 5):
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    total = len(reader.pages)
    pages = min(pages, total)
    for i in range(pages):
        print(f"==== 第{i + 1}页 ====")
        text = reader.pages[i].extract_text()
        if text:
            print(text.strip())
        print("\n")


if __name__ == "__main__":
    pdf_file = r"d:\mathmodel\A题.pdf"
    pages_to_extract = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    extract_first_pages(pdf_file, pages_to_extract)