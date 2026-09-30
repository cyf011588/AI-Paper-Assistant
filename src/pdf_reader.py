from pathlib import Path
import json
import pymupdf



def read_pdf(pdf_path: Path):
    """
    Read PDF and extract text page by page.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )


    doc = pymupdf.open(pdf_path)


    pages = []


    for page_number, page in enumerate(doc):

        text = page.get_text()


        pages.append(
            {
                "page": page_number + 1,
                "text": text
            }
        )


    return pages



def save_to_json(data, output_path: Path):
    """
    Save extracted data into JSON.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )



if __name__ == "__main__":


    # 项目根目录
    BASE_DIR = Path(__file__).resolve().parent.parent


    # PDF路径
    pdf_path = (
        BASE_DIR
        / "data"
        / "papers"
        / "pdf1.pdf"
    )


    # 输出路径
    output_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "pdf1.json"
    )


    pages = read_pdf(pdf_path)


    save_to_json(
        pages,
        output_path
    )


    print("PDF processing completed!")