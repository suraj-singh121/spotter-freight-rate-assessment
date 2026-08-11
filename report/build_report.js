const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun,
} = require("docx");

const PAGE_W = 12240, PAGE_H = 15840; // US Letter, DXA

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 140 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}

function metricsTable(rows) {
  const header = ["Model", "MAE ($)", "RMSE ($)", "MAPE (%)", "R\u00B2"];
  const colWidths = [3400, 1800, 1800, 1800, 1400];
  const mkCell = (text, bold = false) => new TableCell({
    width: { size: colWidths[0], type: WidthType.DXA },
    shading: bold ? { type: ShadingType.CLEAR, fill: "E8EEF0" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold })] })],
  });
  const headerRow = new TableRow({
    children: header.map((t, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "064A56" },
      children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF" })] })],
    })),
  });
  const dataRows = rows.map((r) => new TableRow({
    children: [r.model, r.MAE, r.RMSE, r["MAPE_%"], r.R2].map((v, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      children: [new Paragraph({ children: [new TextRun({ text: String(v) })] })],
    })),
  }));
  return new Table({
    width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

const metrics = JSON.parse(fs.readFileSync("reports/metrics.json", "utf8"));
const chartImage = fs.readFileSync("scorer_results/candidate_december.png");

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Freight Rate Prediction — Assessment Report", bold: true, size: 40, color: "064A56" })],
        spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Machine Learning Engineer take-home · Spotter", italics: true, size: 22, color: "455A60" })],
        spacing: { after: 400 },
      }),

      h1("1. Data Exploration & Quality Issues"),
      p("The development set (data/train_test.csv) contains 48,000 labeled loads from 2025-01-01 through 2025-10-31, with 14 columns: load_id, pickup/delivery city + lat/lon, distance, equipment, weight, date, market_index, quote_signal, and the target posted_rate."),
      h2("Key findings"),
      bullet("posted_rate is strongly driven by distance (Pearson r ≈ 0.91). Weight, market_index and quote_signal individually have weak linear correlation with the raw rate, but do carry information once distance is accounted for (rate-per-mile), and were kept as model features."),
      bullet("Equipment type carries a modest rate premium: Reefer > Flatbed > Dry Van on average."),
      bullet("There is a mild seasonal / weekly pattern in rate-per-mile (slightly higher mid-year and mid-week), used via month, day-of-week and cyclical day-of-year features."),
      bullet("posted_rate is right-skewed with a small tail of very high rate-per-mile loads (~0.7% of rows > 5x the median rate-per-mile). These look like genuine spot-market surge pricing rather than data errors — the associated distances and cities are all valid — so they were kept in training rather than removed, but they do inflate RMSE relative to MAE."),
      h2("Data-quality issues identified and how they were addressed"),
      bullet("weight: ~0.6% of rows (292/48,000) had sign-flipped negative values (e.g. -31,724 lb) whose magnitudes match the normal weight distribution exactly. Fixed with abs(). A further ~0.6% were missing and were imputed using the median weight for that equipment type."),
      bullet("market_index: ~0.8% missing (374/48,000), imputed using the median market_index for that calendar month."),
      bullet("A small number of rows (22/48,000, all for the Shreveport ↔ New Orleans lane) have a distance value inconsistent with the haversine distance implied by their lat/lon. This looks like a geocoding issue for those two cities rather than a distance error. It affects <0.05% of rows and was left as-is (flagged here rather than 'corrected' without ground truth)."),
      bullet("No duplicate load_id or fully duplicated rows were found."),

      h1("2. Train / Test Split & Validation Approach"),
      p("The task is fundamentally a forecasting problem: validation.csv and the December chart both ask for predictions on dates later than most of the labeled data. A random split would let the model be evaluated on dates interleaved with its training data, overstating real-world accuracy."),
      p("Split used: sort all rows by date, train on the earliest 80% (2025-01-01 → 2025-08-31, 38,400 rows), hold out the most recent 20% (2025-09-01 → 2025-10-31, 9,600 rows) purely for evaluation. All reported metrics below are on this time-based holdout, never seen during training."),

      h1("3. Modeling"),
      p("Three model families were compared on the same time-based holdout: linear regression (baseline), random forest, and gradient-boosted trees (XGBoost). A second XGBoost model was trained on a reduced feature set (same features, minus market_index/quote_signal) because those two columns are not available in december_chart_inputs.csv — they represent live market-quote conditions that can't be known in advance for a future, hypothetical date. Using one shared cleaning/feature pipeline (src/data_utils.py) keeps both models consistent."),
      metricsTable(metrics),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      p("XGBoost was selected: lowest MAE and MAPE of the three families, and by a wide margin on typical-case error (MAPE 5.5% vs 10.7% for linear regression). The reduced (no-market-signal) model performs nearly as well (MAE $128.6 vs $124.7), confirming market_index/quote_signal add only a small amount of incremental signal once distance, geography, equipment and date are captured — consistent with their weak standalone correlations in the exploration step."),
      p("Top features by importance (full model): distance, pickup/delivery longitude, delivery latitude, then equipment type. The reduced model relies on the same top features minus the two market columns."),

      h1("4. December 2025 Forecast"),
      p("december_chart_inputs.csv fixes pickup=Lexington, delivery=Fort Wayne, distance=360 mi, equipment=Dry Van, weight=32,000 lb for all 31 days of December 2025 — only the date changes. Lexington/Fort Wayne lat/lon (not present in that file) were looked up from their known values in train_test.csv so the reduced model sees the same feature schema it was trained on. Predictions come from the reduced (no-market-signal) model described above, run once per date."),
      new Paragraph({
        children: [new ImageRun({ data: chartImage, transformation: { width: 620, height: 276 }, type: "png" })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 120 },
      }),
      p("The forecast (~$820–$837, i.e. ~$2.29–$2.32/mile) shows a repeating weekly pattern — modest midweek premium, modest weekend dip — which is the dominant date-driven signal the model found in the training data. Because train_test.csv only spans January–October 2025, December is outside the observed date range; the cyclical (sin/cos) day-of-year encoding lets the model generalize via periodicity (December sits close to January on the yearly cycle) rather than extrapolating linearly, but the resulting range is narrower than a full seasonal swing might realistically be — a limitation worth flagging rather than presenting as high-confidence.", { }),

      h1("5. Reproducing These Results"),
      p("See README.md in the repository root for exact setup and run commands (python src/train.py, then python src/predict.py, then the provided score.py). Full metrics, feature importances and EDA notes are saved under reports/."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("report/Freight_Rate_Assessment_Report.docx", buf);
  console.log("wrote report/Freight_Rate_Assessment_Report.docx");
});
