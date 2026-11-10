"""Build Markdown and a seven-page PDF from measured artifacts and reviewed observations."""
from html import escape
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from config import LOGS, FIGURES
from data import read_json, digest, save_json


def pct(value):
    return f'{100 * value:.2f}%'


def build_sections():
    runs = {k: read_json(LOGS / f'run_{k}.json') for k in 'ABCD'}
    m = read_json(LOGS / 'test_metrics.json')
    a = read_json(LOGS / 'analysis.json')
    s = read_json(LOGS / 'split.json')
    selected = read_json(LOGS / 'selection.json')
    observed = read_json(LOGS / 'observations.json')
    runtime = read_json(LOGS / 'runtime.json')
    winner = runs[selected['id']]
    total_minutes = sum(r['seconds'] for r in runs.values()) / 60
    sections = []

    def section(title, blocks):
        sections.append(dict(title=title, blocks=blocks))

    section('01 / 任务、数据与模型设计', [
        ('p', '当代人工智能作业三：CIFAR-10 图像分类。实验目标是在可复现的数据隔离流程下，比较小型 CNN 的残差结构与数据增强，并解释最终分类错误。所有性能数字来自本机执行记录。'),
        ('p', f'最终模型为配置 {selected["id"]}（{winner["config"]["architecture"]} / {winner["config"]["augmentation"]}），验证准确率 {pct(winner["best_val_accuracy"])}；官方测试集正确 {m["correct"]}/10,000，准确率 {pct(m["accuracy"])}，macro-F1={m["macro_f1"]:.4f}。模型由验证结果选出，测试结果未参与选择。'),
        ('h', '数据与防泄漏设计'),
        ('p', '使用官方 CIFAR-10 Python 版本 [1]。50,000 张训练图像按类别、种子 42 划分为 45,000 张训练和 5,000 张验证，每类分别 4,500/500 张。官方 10,000 张测试图像保持独立，每类 1,000 张。所有输入均为 RGB、32×32。下载包与批次均校验 MD5，并保存 SHA256。'),
        ('p', f'划分后仅用训练子集估计像素通道均值：{", ".join(f"{v:.6f}" for v in s["mean"])}；标准差：{", ".join(f"{v:.6f}" for v in s["std"])}。变换为 (x/255 - mean)/std。验证和测试不使用随机增强，不参与统计量拟合。'),
        ('h', '自行实现的 CNN'),
        ('table', [['阶段', '卷积与操作', '输出形状（不含批次）'],
                   ['输入', 'RGB 标准化', '3×32×32'],
                   ['Block 1', '3×3 Conv: 3→16→16；BN/ReLU；MaxPool', '16×16×16'],
                   ['Block 2', '3×3 Conv: 16→32→32；BN/ReLU；MaxPool', '32×8×8'],
                   ['Block 3', '3×3 Conv: 32→64→64；BN/ReLU；MaxPool', '64×4×4'],
                   ['分类头', 'Global Average Pool；Linear 64→10', '10 logits']]),
        ('p', 'plain 分支每块依次执行 Conv-BN-ReLU-Conv-BN-ReLU-Pool。residual 分支在第二次 BN 后增加 1×1 Conv+BN 的投影支路，然后相加、ReLU、池化；这是针对小图像实现的浅层残差 CNN，不是直接调用标准 ResNet。残差思想参考 [2]，所有权重从头初始化。'),
        ('p', f'plain 有 {runs["A"]["parameters"]:,} 个可训练参数；residual 有 {runs["C"]["parameters"]:,} 个。BatchNorm 和权重衰减用于所有配置，架构变化包含投影支路及其参数开销，不能解释为“完全等参数”比较。'),
    ])
    comparison = [['配置', '架构 / 增强', '最佳轮', '干净训练 Acc', '验证 Acc', '分钟']]
    for k, r in runs.items():
        comparison.append([k, f'{r["config"]["architecture"]} / {r["config"]["augmentation"]}', str(r['best_epoch']),
                           pct(r['clean_train']['accuracy']), pct(r['best_val_accuracy']), f'{r["seconds"]/60:.1f}'])
    effects = '；'.join(f'{x["name"]}：{x["difference_pp"]:+.2f} 个百分点' for x in a['contrasts'])
    section('02 / 实验设置与受控对照', [
        ('p', '四组构成 2×2 因子实验。SGD：初始 lr=0.1，momentum=0.9，weight_decay=0.0005；交叉熵；batch=256；固定 20 轮。前两轮 lr 为 0.05/0.1，此后按余弦衰减至 0.001。无早停、无预训练、无额外数据。'),
        ('p', f'在 AMD Ryzen 5 5600H CPU 上采用 FP32、channels_last、{runtime["threads"]} 个计算线程；线程数仅由合成输入计时选定。数据顺序与增强各用独立随机流，四组共享划分和优化协议。四组训练及最佳检查点训练集评估共用时 {total_minutes:.1f} 分钟，未计下载及环境安装。'),
        ('p', 'crop_flip：反射填充 4 像素、随机裁剪回 32×32、概率 0.5 水平翻转，仅用于训练。最佳检查点按验证 accuracy 降序、loss 升序、轮次升序选择；最终配置继续按参数量、配置 ID 打破平局。'),
        ('table', comparison),
        ('image', ('ablation.png', 490, 190)),
        ('p', '实测单因素差异：' + effects + '。'),
        ('p', '以上是同一划分和种子下的条件性差异。B-A 与 D-C 检验增强在不同架构中的作用；C-A 与 D-B 检验残差在不同增强条件中的作用。D-A 同时改变两个因素，不能单独归因。单次训练的差异不等于统计显著性。'),
        ('p', f'具体而言，残差结构在无增强时的验证差异为 {(runs["C"]["best_val_accuracy"]-runs["A"]["best_val_accuracy"])*100:+.2f} 个百分点，有增强时为 {(runs["D"]["best_val_accuracy"]-runs["B"]["best_val_accuracy"])*100:+.2f} 个百分点，不能据此认为残差总能提高准确率。两种条件下的残差效应相差 {(runs["D"]["best_val_accuracy"]-runs["B"]["best_val_accuracy"]-runs["C"]["best_val_accuracy"]+runs["A"]["best_val_accuracy"])*100:.2f} 个百分点，提示结构收益依赖训练条件，需更多种子验证。'),
        ('p', '余弦调度在所有配置中一致，属于已实现的训练策略；本次未与恒定学习率另做消融，不能把整体提升单独归功于调度。'),
    ])
    gap = winner['clean_train']['accuracy'] - winner['best_val_accuracy']
    section('03 / 训练过程与泛化讨论', [
        ('image', ('learning_curves.png', 490, 535)),
        ('p', '图中训练指标在训练模式、逐批参数更新过程中计算；增强组的训练输入经过随机增强，验证指标来自当轮结束后的固定模型。因此两条曲线的数值差不能直接当作严格的泛化差距。'),
        ('p', f'为建立可比口径，表 02 的训练准确率使用最佳检查点、eval 模式、无增强、完整 45,000 张训练图像。最终模型在此口径下训练 Acc={pct(winner["clean_train"]["accuracy"])}，验证 Acc={pct(winner["best_val_accuracy"])}，差值为 {gap*100:.2f} 个百分点。'),
        ('p', observed['learning']),
    ])
    classes = [['真实类别', '正确 / 样本', '类别准确率', 'Precision', 'F1']]
    for c in m['per_class']:
        classes.append([c['name'], f'{c["correct"]}/{c["support"]}', pct(c['accuracy']), f'{c["precision"]:.3f}', f'{c["f1"]:.3f}'])
    section('04 / 测试集结果与混淆矩阵', [
        ('p', f'冻结的配置 {selected["id"]}、第 {selected["best_epoch"]} 轮检查点在测试集正确 {m["correct"]}/10,000，总体准确率 {pct(m["accuracy"])}，交叉熵 {m["loss"]:.4f}，macro-F1={m["macro_f1"]:.4f}。近似 Wilson 95% 区间为 [{pct(m["wilson_95"][0])}, {pct(m["wilson_95"][1])}]；它只描述有限测试样本的不确定性，不包含重新训练或重新划分的不确定性。'),
        ('image', ('confusion_matrix.png', 400, 326)),
        ('table', classes),
        ('p', '矩阵行是真值、列是预测，每行合计 1,000。每类准确率定义为该类正确数/该类样本数，即该类召回率；总体准确率是全部正确样本/10,000。由于测试类均衡，总体准确率也等于各类准确率的算术平均。'),
    ])
    pairs = '；'.join(f'{r["true"]}→{r["predicted"]} {r["count"]} 张' for r in a['top_confusions'][:5])
    section('05 / 误分类样本分析', [
        ('p', '按预先约定规则，从每个官方标签类别选择预测置信度最高的错误，同分取最小测试索引。图中的 true 指官方标签，不是另行人工复标的结果。这展示的是高置信错误，不是随机抽样，也不代表各类错误频率。图像按原始 32×32 最近邻放大，保留其低分辨率特征。'),
        ('image', ('misclassified.png', 490, 224)),
        ('p', '最常见的有向混淆（实测）：' + pairs + '。'),
        *[('p', line) for line in observed['errors']],
        ('p', '解释边界：上述视觉线索是针对具体图像的解释假设；没有额外标注、遮挡实验或反事实对照，不能证明模型真正依据了某个物体部位。softmax 置信度也不是经过校准的正确概率。'),
    ])
    section('06 / Grad-CAM 与扩展思考', [
        ('p', '对上一节按类别顺序确定的前 4 个误例，解释模型预测类别的 logit。对最后一个卷积层 features.2.conv2 的 8×8 特征图计算梯度空间均值权重，再取加权和、ReLU，归一化并双线性上采样至 32×32 [3]。不反向更新权重。'),
        ('image', ('gradcam.png', 346, 420)),
        *[('p', line) for line in observed['gradcam']],
        ('p', '热图展示目标分数对特征的梯度关联，不是对象分割，也不是因果证明。8×8 分辨率本身很粗，上采样不能补充真实空间信息。测试误例和热图只用于事后分析，未反过来修改模型。'),
        ('p', '后续可在独立实验中比较更长训练预算、多种子、多尺度特征、注意力或校准方法。这些实验需重新注册验证协议，并保留独立测试数据。本次未把这些建议写成已实现的结果。'),
    ])
    section('07 / 复现、局限与参考来源', [
        ('h', '执行与结果证据'),
        ('p', '从代码目录按 README 创建 conda 环境，再运行 run_pipeline.py 的 environment、prepare、verify、train、select、test、analyze 阶段。每阶段经过 check_rigor.py 门禁；训练脚本会输出并读取仓库根及实验的规范文件。完整命令见 README.md。'),
        ('p', 'results/logs/split.json 保存全部 45k/5k 索引及训练统计量；run_A-D.json 保存源码/划分/检查点指纹、参数和最佳轮次；results/tables/history_A-D.csv 保存 80 轮曲线；selection.json 是测试前冻结的选模记录；test_predictions.csv 保存按官方顺序的 10,000 行预测及概率，可独立复算矩阵与准确率。'),
        ('p', '原始 CIFAR-10 和检查点不上传仓库；下载校验和完整训练命令可重新生成。environment.yml 定义 conda 依赖，environment-lock.txt 固定本次 Windows 构建。报告的关键数值由已落盘的 JSON/CSV 自动填入。'),
        ('h', '复核范围与局限'),
        ('p', '正式实验覆盖 4 个配置、每组 20 轮、完整 45,000 张训练子集。独立环境复核记录见 reproduction.json，区分依赖安装、划分/统计量复算、逐样本预测指标复算和短程训练重放；短程重放不代表在第二环境重训了完整 80 轮。'),
        ('p', '本实验只使用一个种子和一次划分，验证集被用于选择架构和轮次，验证结果存在选择偏差。测试评估的是选出的单个模型；其他配置的测试成绩没有计算。小型网络、CPU 预算与 20 轮训练限制了可达到的性能，不能声称最优架构或充分收敛。'),
        ('p', '课程评分包括实现、性能、报告和创新，题面没有给出准确率到分数的映射。本报告不预测得分。GitHub 是本次交付位置；课程 PDF 指定的坚果云提交渠道仍应单独处理。'),
        ('h', '参考来源'),
        ('p', '[1] Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. 数据集说明与校验值：https://www.cs.toronto.edu/~kriz/cifar.html'),
        ('p', '[2] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385'),
        ('p', '[3] Selvaraju, R. R., et al. (2016/2019). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. https://arxiv.org/abs/1610.02391'),
        ('p', '[4] PyTorch 官方教程：Training a Classifier；实际执行版本 PyTorch 2.5.1 / torchvision 0.20.1。https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html'),
        ('p', '[5] 课程材料：《当代人工智能作业三——图像分类》，2026-11-05，6 页；任务第 5 页、评分第 6 页。'),
    ])
    return sections


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
    normal = ParagraphStyle('body', fontName=font_name, fontSize=10, leading=15,
                            textColor=colors.HexColor('#203542'), wordWrap='CJK', spaceAfter=8)
    title = ParagraphStyle('title', parent=normal, fontSize=18, leading=24, spaceAfter=17)
    heading = ParagraphStyle('heading', parent=normal, fontSize=12, leading=18, spaceBefore=6, spaceAfter=8)
    cell = ParagraphStyle('cell', parent=normal, fontSize=9, leading=13, spaceAfter=0)
    out = Path('output/pdf/Lab3_report.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    story, markdown = [], ['# 当代人工智能作业三：CIFAR-10 图像分类\n']
    for i, section in enumerate(sections):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(escape(section['title']), title))
        markdown.append('## ' + section['title'] + '\n')
        for kind, value in section['blocks']:
            if kind in ['p', 'h']:
                story.append(Paragraph(escape(value), heading if kind == 'h' else normal))
                markdown.append(('### ' if kind == 'h' else '') + value + '\n')
            elif kind == 'image':
                name, max_width, max_height = value
                image = Image(str(FIGURES / name))
                scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
                image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
                image.hAlign = 'CENTER'
                story.extend([image, Spacer(1, 10)])
                markdown.append(f'![{name.removesuffix(".png")}](results/figures/{name})\n')
            elif kind == 'table':
                if len(value[0]) == 3:
                    widths = [56, 310, 128]
                elif len(value[0]) == 6:
                    widths = [30, 174, 44, 84, 84, 68]
                else:
                    widths = [115, 105, 100, 82, 82]
                rows = [[Paragraph(escape(str(v)), cell) for v in row] for row in value]
                table = Table(rows, colWidths=widths, repeatRows=1, hAlign='CENTER')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5EDF1')),
                    ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#91A4AF')),
                    ('LINEBELOW', (0, -1), (-1, -1), .7, colors.HexColor('#91A4AF')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 10)])
                markdown.append('| ' + ' | '.join(value[0]) + ' |')
                markdown.append('| ' + ' | '.join(['---'] * len(value[0])) + ' |')
                markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in value[1:])
                markdown.append('')

    def footer(canvas, doc):
        canvas.setTitle('当代人工智能作业三：CIFAR-10 图像分类')
        canvas.setAuthor('AI-Lab')
        canvas.setStrokeColor(colors.HexColor('#C5D0D6'))
        canvas.line(44, 37, A4[0]-44, 37)
        canvas.setFont(font_name, 9)
        canvas.setFillColor(colors.HexColor('#526875'))
        canvas.drawString(44, 24, 'AI-Lab / 作业三 / 实验结果与分析')
        canvas.drawRightString(A4[0]-44, 24, f'{doc.page}')

    document = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=44, rightMargin=44,
                                 topMargin=40, bottomMargin=48)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    Path('REPORT.md').write_text('\n'.join(markdown), encoding='utf-8')
    save_json(LOGS / 'report_manifest.json', dict(pdf_sha256=digest(out), markdown_sha256=digest('REPORT.md'),
              section_count=len(sections), font=font_name, embedded_cjk_font=font_path is not None,
              inputs={p.name: digest(p) for p in LOGS.glob('*.json')
              if p.name in ['test_metrics.json', 'analysis.json', 'observations.json', 'selection.json', 'image_label_audit.json'] or p.name.startswith('run_')}))
    print('Built REPORT.md and output/pdf/Lab3_report.pdf', flush=True)


if __name__ == '__main__':
    render(build_sections())
