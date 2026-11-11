"""Build the Markdown and PDF versions of the report from the recorded experiment."""
from html import escape
from io import BytesIO
import os
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from check_rigor import REPOSITORY_URL, validate_report_format
from config import FIGURES, LOGS
from data import digest, read_json, save_json

TITLE = '当代人工智能作业三 · CIFAR-10 图像分类'


def pct(value):
    return f'{100 * value:.2f}%'


def build_sections():
    runs = {k: read_json(LOGS / f'run_{k}.json') for k in 'ABCD'}
    m = read_json(LOGS / 'test_metrics.json')
    analysis = read_json(LOGS / 'analysis.json')
    split = read_json(LOGS / 'split.json')
    selected = read_json(LOGS / 'selection.json')
    observed = read_json(LOGS / 'observations.json')
    runtime = read_json(LOGS / 'runtime.json')
    winner = runs[selected['id']]
    val = {k: r['best_val_accuracy'] for k, r in runs.items()}
    gap = {k: 100 * (r['clean_train']['accuracy'] - val[k]) for k, r in runs.items()}

    architecture = [
        ['阶段', '操作', '输出（通道×高×宽）'],
        ['卷积块 1', '3×3 卷积，通道 3→16→16，再池化', '16×16×16'],
        ['卷积块 2', '3×3 卷积，通道 16→32→32，再池化', '32×8×8'],
        ['卷积块 3', '3×3 卷积，通道 32→64→64，再池化', '64×4×4'],
        ['分类头', '全局平均池化，64 维向量映射至 10 类', '10 个 logits'],
    ]
    comparison = [['配置', '架构 / 增强', '最佳轮', '训练准确率¹', '验证准确率', '用时（分）']]
    for k, run in runs.items():
        comparison.append([k, f'{run["config"]["architecture"]} / {run["config"]["augmentation"]}',
                           str(run['best_epoch']), pct(run['clean_train']['accuracy']),
                           pct(val[k]), f'{run["seconds"] / 60:.1f}'])
    classes = [['真实类别', '正确数 / 样本数', '逐类准确率', '精确率', 'F1']]
    for c in m['per_class']:
        classes.append([c['name'], f'{c["correct"]} / {c["support"]}', pct(c['accuracy']),
                        f'{c["precision"]:.3f}', f'{c["f1"]:.3f}'])
    errors_by_class = {c['name']: c['support'] - c['correct'] for c in m['per_class']}
    pairs = [['真实类别', '预测类别', '误分张数', '占该真实类别全部错误']]
    for pair in analysis['top_confusions'][:5]:
        source, n = pair['true'], pair['count']
        pairs.append([source, pair['predicted'], str(n), pct(n / errors_by_class[source])])

    return [
        ('数据、模型与训练方案', [
            ('p', f'在 CIFAR-10 上比较普通 CNN 与残差 CNN，考察随机裁剪和翻转能否改善泛化。'
                  f'验证集选出的配置 {selected["id"]} 在官方测试集上正确分类 {m["correct"]}/10,000 张图像，准确率为 {pct(m["accuracy"])}。'),
            ('h', '数据划分与预处理'),
            ('p', '输入为 32×32 RGB 图像。以种子 42 对官方训练集逐类抽样，每类取 4,500 张训练、500 张验证，共 45,000 / 5,000 张。'
                  '官方测试集含 10,000 张图像，仅在冻结最终模型后评估。'),
            ('p', f'将像素缩放至 [0, 1]，按训练子集的 RGB 均值（{", ".join(f"{v:.4f}" for v in split["mean"])}）'
                  f'和标准差（{", ".join(f"{v:.4f}" for v in split["std"])}）统一标准化。增强组仅在训练时反射填充 4 像素、随机裁回 32×32，并以 0.5 概率水平翻转。'),
            ('h', '网络与优化'),
            ('table', architecture),
            ('p', f'普通 CNN 的每块含两次卷积与 BatchNorm，ReLU 激活后以 2×2 最大池化降采样。'
                  f'残差 CNN 在第二次 BatchNorm 后加入 1×1 卷积与 BatchNorm 的投影支路，相加后再激活、池化。'
                  f'两种网络均从头训练，参数量分别为 {runs["A"]["parameters"]:,} 和 {runs["C"]["parameters"]:,}。'),
            ('p', f'四组共用划分与训练预算。用交叉熵、SGD、动量 0.9、权重衰减 0.0005、批大小 256 训练 20 轮。'
                  f'前两轮学习率为 0.05、0.1，随后余弦衰减至 0.001。每轮结束在验证集评估，按准确率最高、损失最低、轮次最早保留检查点。'
                  f'实测采用 Ryzen 5 5600H CPU、FP32 和 {runtime["threads"]} 个计算线程。'),
            ('image', ('ablation.png', 482, 175, f'图 1　加入增强后，普通和残差 CNN 的验证准确率分别提高 {(val["B"]-val["A"])*100:.2f}、{(val["D"]-val["C"])*100:.2f} 个百分点。'
                      f'换成残差结构后，无增强时变化为 {(val["C"]-val["A"])*100:+.2f}、有增强时为 {(val["D"]-val["B"])*100:+.2f} 个百分点。')),
        ]),
        ('四组对照与训练曲线', [
            ('table', comparison),
            ('p', '¹ 训练准确率以各组最佳检查点在完整、无增强的训练集上测得。图 2 的训练曲线记录训练模式下逐批更新时的在线指标。'),
            ('image', ('learning_curves.png', 455, 480, '图 2　四组各 20 轮的交叉熵和准确率。每行对应一组，横轴为轮次，两条线分别为训练集与验证集。')),
            ('p', f'A、C 的训练与验证准确率分别相差 {gap["A"]:.2f}、{gap["C"]:.2f} 个百分点，B、D 的差距缩至 {gap["B"]:.2f}、{gap["D"]:.2f} 个百分点。'
                  + observed['learning']),
        ]),
        ('测试集表现与逐类准确率', [
            ('p', f'配置 {selected["id"]} 第 {selected["best_epoch"]} 轮的验证准确率为 {pct(winner["best_val_accuracy"])}。'
                  f'冻结后在官方测试集上正确分类 {m["correct"]}/10,000 张，总准确率 {pct(m["accuracy"])}，交叉熵 {m["loss"]:.4f}，'
                  f'macro-F1 为 {m["macro_f1"]:.4f}。准确率的 Wilson 95% 区间为 {pct(m["wilson_95"][0])} 至 {pct(m["wilson_95"][1])}，反映测试样本的抽样不确定性。'),
            ('image', ('confusion_matrix.png', 442, 352, '图 3　混淆矩阵。行是真实类别，列是预测类别，单元格给出样本数。')),
            ('table', classes),
            ('p', '每类准确率等于该类正确数除以该类样本数，即召回率。十类各有 1,000 张，总准确率等于十个逐类准确率的平均值。'
                  '汽车最易识别，猫最难识别。下一页用完整混淆计数与具体图像分析误差。'),
        ]),
        ('误分类样本与常见混淆', [
            ('h', '数量最多的五种有向错误'),
            ('table', pairs),
            ('p', observed['errors'][0]),
            ('h', '每一类中最有把握的错误'),
            ('image', ('misclassified.png', 493, 222, '图 4　每个官方真实类别选取预测置信度最高的错误，同分取最小测试索引。图中标出索引、标签、预测和置信度。')),
            *[('p', line) for line in observed['errors'][1:]],
        ]),
        ('Grad-CAM 解释与复现', [
            ('p', '对图 4 按类别顺序排列的前四个误例，取预测类别的 logit 对最后一层卷积的 8×8 特征图求梯度。'
                  '将梯度空间均值作为通道权重，加权求和、取 ReLU 并归一化，双线性上采样至 32×32。'),
            ('image', ('gradcam.png', 340, 365, '图 5　从左至右依次为带官方标签的图像、预测类别的 Grad-CAM 热图、热图与原图的叠加。')),
            *[('p', line) for line in observed['gradcam']],
            ('h', '实验局限与复现'),
            ('p', '单次划分和种子限制了比较的精度。残差投影增加了参数量，余弦调度由四组共享。后续可固定参数预算、重复多个种子，检验有增强时的残差收益。'),
            ('p', '按仓库 README 创建 conda 环境后，运行 python reproduce.py --mode full 可重做四组实验。'
                  'results/tables/ 保存 80 轮训练记录及 10,000 行测试预测。独立环境已复算划分、标准化统计量和预测指标，并重放四步合成数据训练。'),
            ('h', '参考资料'),
            ('links', [('CIFAR-10 数据集', 'https://www.cs.toronto.edu/~kriz/cifar.html'),
                       ('He 等 · 残差网络', 'https://arxiv.org/abs/1512.03385'),
                       ('Selvaraju 等 · Grad-CAM', 'https://arxiv.org/abs/1610.02391'),
                       ('PyTorch 分类教程', 'https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html')]),
        ]),
    ]


def render(sections):
    # Embed a CJK font when available; never publish the system font file itself.
    candidates = []
    if os.environ.get('LAB3_REPORT_FONT'):
        candidates.append(Path(os.environ['LAB3_REPORT_FONT']))
    if os.environ.get('WINDIR'):
        candidates.append(Path(os.environ['WINDIR']) / 'Fonts' / 'msyh.ttc')
    font_path = next((p for p in candidates if p.is_file()), None)
    font_name = 'Lab3CJK' if font_path else 'STSong-Light'
    if font_path:
        pdfmetrics.registerFont(TTFont(font_name, str(font_path), subfontIndex=0))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    normal = ParagraphStyle('body', fontName=font_name, fontSize=10.2, leading=15.4,
                            textColor=colors.HexColor('#203542'), wordWrap='CJK', spaceAfter=7)
    title = ParagraphStyle('title', parent=normal, fontSize=21, leading=28, spaceAfter=10)
    section_title = ParagraphStyle('section', parent=normal, fontSize=17, leading=23, spaceAfter=10)
    heading = ParagraphStyle('heading', parent=normal, fontSize=12, leading=17, spaceBefore=3,
                             spaceAfter=5, textColor=colors.HexColor('#22586B'))
    caption = ParagraphStyle('caption', parent=normal, fontSize=9.1, leading=13.3, spaceBefore=3,
                             spaceAfter=9, textColor=colors.HexColor('#415863'))
    cell = ParagraphStyle('cell', parent=normal, fontSize=9.4, leading=13.4, spaceAfter=0)
    widths = {3: [91, 248, 152], 4: [109, 109, 88, 185],
              5: [104, 130, 99, 83, 75], 6: [42, 152, 52, 82, 86, 77]}

    def link(label, url):
        return f'<link href="{escape(url, quote=True)}" color="#0A5976"><u>{escape(label)}</u></link>'

    story = [Paragraph(escape(TITLE), title),
             Paragraph('GitHub 仓库　' + link(REPOSITORY_URL, REPOSITORY_URL), normal)]
    markdown = [f'# {TITLE}\n', f'GitHub 仓库　[{REPOSITORY_URL}]({REPOSITORY_URL})\n']
    for i, (name, blocks) in enumerate(sections):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(escape(name), section_title))
        markdown.append(f'## {name}\n')
        for kind, value in blocks:
            if kind in ['p', 'h']:
                story.append(Paragraph(escape(value), heading if kind == 'h' else normal))
                markdown.append(('### ' if kind == 'h' else '') + value + '\n')
            elif kind == 'image':
                filename, max_width, max_height, description = value
                image = Image(str(FIGURES / filename))
                scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
                image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
                image.hAlign = 'CENTER'
                story.extend([image, Paragraph(escape(description), caption)])
                markdown.append(f'![{description}](results/figures/{filename})\n\n{description}\n')
            elif kind == 'table':
                rows = [[Paragraph(escape(v), cell) for v in row] for row in value]
                table = Table(rows, colWidths=widths[len(value[0])], repeatRows=1, hAlign='CENTER')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E1EDF0')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F8F9')]),
                    ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#91A4AF')),
                    ('LINEBELOW', (0, -1), (-1, -1), .7, colors.HexColor('#91A4AF')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ]))
                story.extend([table, Spacer(1, 7)])
                markdown.append('| ' + ' | '.join(value[0]) + ' |')
                markdown.append('| ' + ' | '.join(['---'] * len(value[0])) + ' |')
                markdown.extend('| ' + ' | '.join(row) + ' |' for row in value[1:])
                markdown.append('')
            elif kind == 'links':
                story.append(Paragraph('　 /　'.join(link(label, url) for label, url in value), caption))
                markdown.append(' · '.join(f'[{label}]({url})' for label, url in value) + '\n')

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#C5D0D6'))
        canvas.line(44, 37, A4[0] - 44, 37)
        canvas.setFont(font_name, 9)
        canvas.setFillColor(colors.HexColor('#526875'))
        canvas.drawString(44, 24, 'AI-Lab / 作业三 / CIFAR-10')
        canvas.drawRightString(A4[0] - 44, 24, str(doc.page))

    stream = BytesIO()
    document = SimpleDocTemplate(stream, pagesize=A4, leftMargin=44, rightMargin=44,
                                 topMargin=37, bottomMargin=48, title=TITLE, author='AI-Lab', invariant=1)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    report = '\n'.join(markdown)
    pdf = PdfReader(stream)
    validate_report_format(report, pdf)

    out = Path('output/pdf/Lab3_report.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(stream.getvalue())
    Path('REPORT.md').write_text(report, encoding='utf-8')
    inputs = ['analysis', 'image_label_audit', 'observations', 'runtime', 'selection', 'split', 'test_metrics']
    inputs.extend(f'run_{k}' for k in 'ABCD')
    save_json(LOGS / 'report_manifest.json', dict(pdf_sha256=digest(out), markdown_sha256=digest('REPORT.md'),
              page_count=len(pdf.pages), repository_url=REPOSITORY_URL, section_count=len(sections),
              font=font_name, embedded_cjk_font=font_path is not None,
              inputs={f'{name}.json': digest(LOGS / f'{name}.json') for name in inputs}))
    print(f'Built REPORT.md and {out.as_posix()} ({len(pdf.pages)} pages)', flush=True)


if __name__ == '__main__':
    render(build_sections())
