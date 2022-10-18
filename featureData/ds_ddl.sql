CREATE TABLE "ds_clicks" (
"offer_id" INTEGER,
  "clicked_at" TIMESTAMP
);
CREATE TABLE "ds_leads" (
"lead_uuid" TEXT,
  "requested" REAL,
  "loan_purpose" TEXT,
  "credit" TEXT,
  "annual_income" REAL
);
CREATE TABLE "ds_offers" (
"lead_uuid" TEXT,
  "offer_id" INTEGER,
  "apr" REAL,
  "lender_id" INTEGER
);
