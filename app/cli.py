"""Usage:
    python -m app.cli ingest
    python -m app.cli search "who is eligible for the scholarship?"
"""
import argparse

from app.config import RAW_DIR, TOP_K


def cmd_ingest(_args):
    from app.ingest import parse_pdf
    from app.vectorstore import add_chunks

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Add some and run again.")
        return
    for pdf in pdfs:
        chunks = parse_pdf(pdf)
        add_chunks(chunks)
        tables = sum(c.kind == "table" for c in chunks)
        print(f"{pdf.name}: {len(chunks)} chunks ({tables} tables)")


def cmd_search(args):
    from app.vectorstore import search

    for i, h in enumerate(search(args.query, args.k), start=1):
        print(f"\n#{i}  score={h['score']}  {h['source']}  p.{h['page']}  [{h['kind']}]")
        print(h["text"][:400])


def main():
    parser = argparse.ArgumentParser(prog="schemelens")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=TOP_K)
    s.set_defaults(func=cmd_search)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
