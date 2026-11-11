"""Exercise the report checks against the delivered files and in-memory negative controls."""
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from check_rigor import (REPOSITORY_URL, REPORT_MAX_PAGES, read_json, sha256,
                         validate_report_claims, validate_report_format)


def sample_pdf(pages, url=REPOSITORY_URL, link_page=1, top=56):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=A4)
    for page in range(1, pages + 1):
        y = A4[1] - top
        canvas.drawString(42, y, REPOSITORY_URL)
        if page == link_page and url:
            canvas.linkURL(url, (42, y - 2, 270, y + 12))
        canvas.showPage()
    canvas.save()
    return PdfReader(stream)


def run():
    logs = Path('results/logs')
    report = Path('REPORT.md')
    pdf_file = Path('output/pdf/Lab3_report.pdf')
    text = report.read_text(encoding='utf-8')
    pdf = PdfReader(pdf_file)
    metrics = read_json(logs / 'test_metrics.json')
    selection = read_json(logs / 'selection.json')
    runs = [read_json(logs / f'run_{k}.json') for k in 'ABCD']
    validate_report_format(text, pdf)
    validate_report_claims(text, metrics, selection, runs)
    validate_report_claims('\n'.join(p.extract_text() for p in pdf.pages), metrics, selection, runs)
    for n in (1, REPORT_MAX_PAGES):
        validate_report_format(text, sample_pdf(n))

    controls = []

    def reject(name, expected, validator, *args):
        try:
            validator(*args)
        except AssertionError as exc:
            if expected not in str(exc):
                raise
            controls.append(dict(mutation=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError(f'Accepted invalid report: {name}')

    for name, token, reason in [
        ('overall_accuracy', f'{metrics["accuracy"]:.2%}', 'overall accuracy'),
        ('correct_count', f'{metrics["correct"]}/10,000', 'correct count'),
        ('macro_f1', f'{metrics["macro_f1"]:.4f}', 'macro F1'),
        ('class_accuracy', f'{metrics["per_class"][1]["accuracy"]:.2%}', 'class accuracy'),
        ('validation_accuracy', f'{selection["best_val_accuracy"]:.2%}', 'validation accuracy'),
        ('selected_model', f'配置 {selection["id"]}', 'selected model'),
    ]:
        reject(name, reason, validate_report_claims, text.replace(token, '×'), metrics, selection, runs)

    reject('six_pages', 'pages', validate_report_format, text, sample_pdf(REPORT_MAX_PAGES + 1))
    for name, fixture in [('plain_pdf_url', sample_pdf(1, url=None)),
                          ('wrong_pdf_url', sample_pdf(1, url='https://example.org')),
                          ('pdf_link_on_second_page', sample_pdf(2, link_page=2)),
                          ('pdf_link_at_bottom', sample_pdf(1, top=700))]:
        reject(name, 'PDF opening', validate_report_format, text, fixture)
    reject('plain_markdown_url', 'Markdown opening', validate_report_format, f'# Report\n\n{REPOSITORY_URL}', pdf)
    reject('late_markdown_link', 'Markdown opening', validate_report_format, '# Report\n\nResults\n' + text, pdf)

    result = dict(passed=True, report_sha256=sha256(report), pdf_sha256=sha256(pdf_file),
                  positive_control='Delivered Markdown and PDF accepted; one- and five-page fixtures accepted.',
                  negative_controls=controls, scope='In-memory mutations of report claims, links and pagination.')
    (logs / 'report_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Report accepted; {len(controls)} invalid variants rejected.')


if __name__ == '__main__':
    run()
