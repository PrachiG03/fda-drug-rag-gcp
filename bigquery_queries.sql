
-- FDA Drug Intelligence — BigQuery Analytics Queries
-- Project: healthcare-rag-prachi | Dataset: healthcare_rag
-- Author: Prachi Gehlot


-- QUERY 1: Dataset Overview 
-- Basic stats across all 10 therapeutic categories
SELECT
  search_category,
  COUNT(*)                                          AS total_drugs,
  SUM(has_black_box)                                AS black_box_count,
  ROUND(AVG(warning_length), 0)                     AS avg_warning_length,
  ROUND(AVG(interaction_length), 0)                 AS avg_interaction_length,
  ROUND(SUM(has_black_box) / COUNT(*) * 100, 2)     AS black_box_pct,
  SUM(is_prescription)                              AS prescription_count,
  SUM(CASE WHEN is_prescription = 0 THEN 1 END)    AS otc_count
FROM healthcare_rag.drug_labels
GROUP BY search_category
ORDER BY total_drugs DESC;


-- QUERY 2: CTE- Risk Scoring 
-- Composite risk score per drug using CTE
WITH base_features AS (
  SELECT
    brand_name,
    generic_name,
    manufacturer,
    search_category,
    product_type,
    warning_length,
    interaction_length,
    has_black_box,
    has_interactions,
    is_prescription
  FROM healthcare_rag.drug_labels
  WHERE generic_name IS NOT NULL AND generic_name != ''
),
risk_scored AS (
  SELECT *,
    (has_black_box * 50)
    + CASE
        WHEN warning_length > 3000 THEN 30
        WHEN warning_length > 1500 THEN 15
        ELSE 5
      END
    + CASE
        WHEN interaction_length > 1000 THEN 20
        WHEN interaction_length > 500  THEN 10
        ELSE 0
      END AS risk_score
  FROM base_features
)
SELECT *,
  CASE
    WHEN risk_score >= 70 THEN 'High Risk'
    WHEN risk_score >= 40 THEN 'Medium Risk'
    ELSE 'Low Risk'
  END AS risk_tier
FROM risk_scored
ORDER BY risk_score DESC
LIMIT 20;


-- QUERY 3: Window Functions 
-- Rank drugs by warning severity within each category
-- Shows: ROW_NUMBER, RANK, AVG OVER, NTILE
SELECT
  brand_name,
  generic_name,
  search_category,
  warning_length,
  interaction_length,
  has_black_box,

  -- Rank by warning length within category
  ROW_NUMBER() OVER (
    PARTITION BY search_category
    ORDER BY warning_length DESC
  ) AS rank_in_category,

  -- Overall percentile tier (1 = most severe)
  NTILE(4) OVER (
    ORDER BY warning_length DESC
  ) AS severity_quartile,

  -- Category average for comparison
  ROUND(AVG(warning_length) OVER (
    PARTITION BY search_category
  ), 0) AS category_avg_warning,

  -- How this drug compares to its category average
  warning_length - AVG(warning_length) OVER (
    PARTITION BY search_category
  ) AS vs_category_avg,

  -- Running count per manufacturer
  COUNT(*) OVER (
    PARTITION BY manufacturer
  ) AS manufacturer_drug_count

FROM healthcare_rag.drug_labels
WHERE generic_name IS NOT NULL AND generic_name != ''
ORDER BY search_category, rank_in_category;


-- QUERY 4: JOIN Drug Labels & ML Predictions 
-- Join main table with ML results table
-- Shows black box prediction accuracy per category
SELECT
  d.search_category,
  COUNT(*)                                              AS total_drugs,
  SUM(d.has_black_box)                                  AS actual_black_box,
  SUM(m.black_box_predicted)                            AS predicted_black_box,
  ROUND(AVG(m.black_box_probability) * 100, 2)          AS avg_risk_probability_pct,
  ROUND(SUM(d.has_black_box) / COUNT(*) * 100, 2)       AS actual_black_box_rate_pct,
  ROUND(SUM(m.black_box_predicted) / COUNT(*) * 100, 2) AS predicted_black_box_rate_pct
FROM healthcare_rag.drug_labels d
INNER JOIN healthcare_rag.drug_labels_ml m
  ON d.set_id = m.set_id
GROUP BY d.search_category
ORDER BY avg_risk_probability_pct DESC;


-- QUERY 5: VIEW: Drug Risk Dashboard 
-- Save as view: vw_drug_risk_dashboard
-- Used by Power BI for risk analysis page
CREATE OR REPLACE VIEW healthcare_rag.vw_drug_risk_dashboard AS
WITH scored AS (
  SELECT
    d.set_id,
    COALESCE(NULLIF(d.brand_name, ''), d.generic_name) AS display_name,
    d.generic_name,
    d.manufacturer,
    d.search_category,
    d.product_type,
    d.warning_length,
    d.interaction_length,
    d.has_black_box,
    d.has_interactions,
    d.is_prescription,
    m.black_box_probability,
    m.cluster,
    (d.has_black_box * 50)
    + CASE WHEN d.warning_length > 3000 THEN 30
           WHEN d.warning_length > 1500 THEN 15
           ELSE 5 END
    + CASE WHEN d.interaction_length > 1000 THEN 20
           WHEN d.interaction_length > 500  THEN 10
           ELSE 0 END AS risk_score
  FROM healthcare_rag.drug_labels d
  LEFT JOIN healthcare_rag.drug_labels_ml m ON d.set_id = m.set_id
  WHERE d.generic_name IS NOT NULL AND d.generic_name != ''
)
SELECT *,
  CASE
    WHEN risk_score >= 70 THEN 'High Risk'
    WHEN risk_score >= 40 THEN 'Medium Risk'
    ELSE 'Low Risk'
  END AS risk_tier,
  RANK() OVER (
    PARTITION BY search_category
    ORDER BY risk_score DESC
  ) AS rank_in_category
FROM scored;


-- QUERY 6: Manufacturer Intelligence 
-- Top manufacturers by drug count + black box rate
-- Shows: aggregation, HAVING, calculated metrics
SELECT
  manufacturer,
  COUNT(*)                                          AS total_drugs,
  COUNT(DISTINCT search_category)                   AS categories_covered,
  SUM(has_black_box)                                AS black_box_drugs,
  ROUND(SUM(has_black_box)/COUNT(*)*100, 2)         AS black_box_rate_pct,
  ROUND(AVG(warning_length), 0)                     AS avg_warning_length,
  SUM(is_prescription)                              AS prescription_drugs,
  SUM(CASE WHEN is_prescription = 0 THEN 1 END)    AS otc_drugs
FROM healthcare_rag.drug_labels
WHERE manufacturer IS NOT NULL AND manufacturer != ''
GROUP BY manufacturer
HAVING COUNT(*) >= 10
ORDER BY total_drugs DESC
LIMIT 15;

