import sys

from render_campaign_evidence import main as render_campaign_evidence_main


def main(argv: list[str] | None = None) -> int:
    return render_campaign_evidence_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
