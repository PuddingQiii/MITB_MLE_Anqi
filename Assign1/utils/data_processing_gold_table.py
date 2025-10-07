import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import os
import pyspark.sql.functions as F

def build_gold_feature_store(
    snapshot_date_str: str,
    silver_clickstream_dir: str,
    silver_attributes_dir: str,
    silver_financials_dir: str,
    label_store_dir: str,
    gold_feature_store_dir: str,
    spark
):
    key = snapshot_date_str.replace("-", "_")

    cs_path   = os.path.join(silver_clickstream_dir, f"silver_feature_clickstream_{key}.parquet")
    attr_path = os.path.join(silver_attributes_dir,  f"silver_feature_attributes_{key}.parquet")
    fin_path  = os.path.join(silver_financials_dir,  f"silver_feature_financials_{key}.parquet")

    cs   = spark.read.parquet(cs_path)
    attr = spark.read.parquet(attr_path)
    fin  = spark.read.parquet(fin_path)

    def prep(df):
        return (df
            .withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
            .withColumn("snapshot_date", F.to_date("snapshot_date"))
        )

    cs, attr, fin = map(prep, [cs, attr, fin])

    # 1) Feature engineering in the Gold layer
    # 1.1 Clickstream: Calculate the current total sum of all columns starting with fe_
    fe_cols = [c for c in cs.columns if c.startswith("fe_")]
    if fe_cols:
        expr_sum = None
        for c in fe_cols:
            expr_sum = F.col(c) if expr_sum is None else expr_sum + F.col(c)
        cs = cs.withColumn("click_sum", expr_sum.cast("double"))

    # 1.2 Cast `Annual_Income` to double type.
    if "Annual_Income" in fin.columns:
        fin = fin.withColumn("Annual_Income", F.col("Annual_Income").cast("double"))

    # 2) Label extraction
    lbl_path = os.path.join(label_store_dir, f"gold_label_store_{key}.parquet")
    labels = (spark.read.parquet(lbl_path)
        .withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
        .withColumn("snapshot_date", F.to_date("snapshot_date"))
        .select("Customer_ID", "snapshot_date", "label", "label_def")
    )

    # 3) Join
    joined = (labels
        .join(cs,   on=["Customer_ID", "snapshot_date"], how="left")
        .join(attr, on=["Customer_ID", "snapshot_date"], how="left")
        .join(fin,  on=["Customer_ID", "snapshot_date"], how="left")
    )

    # 4) Revenue per Clicks
    if "Annual_Income" in joined.columns and "click_sum" in joined.columns:
        joined = joined.withColumn(
            "income_per_click",
            F.when((F.col("click_sum").isNotNull()) & (F.col("click_sum") != 0),
                   F.col("Annual_Income") / F.col("click_sum"))
             .otherwise(None)
        )

    # Deduplication + basic alignment.
    joined = joined.dropDuplicates(["Customer_ID", "snapshot_date"])

    # 5) Gold Feature Store
    os.makedirs(gold_feature_store_dir, exist_ok=True)
    out_path = os.path.join(gold_feature_store_dir, f"gold_feature_store_{key}.parquet")
    joined.write.mode("overwrite").parquet(out_path)
    print("Saved feature store:", out_path)

    return joined