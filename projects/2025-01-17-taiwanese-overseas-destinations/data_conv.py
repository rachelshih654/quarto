import pandas as pd

data = pd.read_csv('rawdata.csv')

df = data.apply(lambda x: x.ffill(axis=0), axis=1).copy()

# 設定表頭
header = df.iloc[0].fillna("") + "_" + df.iloc[2].fillna("")
df.columns = header

# 刪除表頭和最後一列
df = df.iloc[3:-1, :-1]
df.reset_index(drop=True, inplace=True)

# 增加區域欄位
df["REGION"] = df["_Residence"].copy()

# 不是地區的名稱改成空值
df["REGION"] = df["REGION"].apply(lambda x: x if x in ["亞洲地區(Asia)", "美洲地區(Americas)", "歐洲地區(Europe)", "大洋洲地區(Oceania)" ,"非洲地區(Africa)", "其他未列明(Unknow)"] else None)

# 往上補齊
df["REGION"] = df["REGION"].ffill(axis=0)

# 把_Residence是地區名稱的資料改成 "全部(ALL)"
df["_Residence"] = df["_Residence"].apply(lambda x: "全部(ALL)" if x in ["亞洲地區(Asia)", "美洲地區(Americas)", "歐洲地區(Europe)", "大洋洲地區(Oceania)" ,"非洲地區(Africa)", "其他未列明(Unknow)"] else x)

# 將整個 DataFrame 中除了特定欄位的數據轉換為數值型別
exclude_columns = ["_Years", "_Months", "REGION", "_Residence"]  # 要排除的欄位
numeric_columns = df.columns.difference(exclude_columns)  # 獲取需要轉換的欄位
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')  # 轉換為數值型別

# 根據 _Years, _Months, REGION 建立新資料的框架
additional_rows = df[["_Years", "_Months", "REGION"]].drop_duplicates()
additional_rows["_Residence"] = "其他(Other)"

# 定義計算邏輯：對於 "其他"，值為 "全部" - 非 "全部" 的加總
def calculate_residual(row, col):
    total_value = df[
        (df["_Years"] == row["_Years"]) &
        (df["_Months"] == row["_Months"]) &
        (df["REGION"] == row["REGION"]) &
        (df["_Residence"] == "全部(ALL)")
    ][col].values[0]
    
    non_total_sum = df[
        (df["_Years"] == row["_Years"]) &
        (df["_Months"] == row["_Months"]) &
        (df["REGION"] == row["REGION"]) &
        (df["_Residence"] != "全部(ALL)") &
        (df["_Residence"] != "其他(Other)")
    ][col].sum()
    
    return total_value - non_total_sum

# 更新 '其他' 的欄位值
for col in df.columns:
    if col not in ["_Years", "_Months", "REGION", "_Residence"]:
        additional_rows[col] = additional_rows.apply(
            lambda row: calculate_residual(row, col)
            if row["_Residence"] == "其他(Other)" else None,
            axis=1
        )

# 合併新增資料到原始資料
data = pd.concat([df, additional_rows], ignore_index=True)

# 將數據從寬表轉換為長表
melted_df = pd.melt(
    data, 
    id_vars=["_Years", "_Months", "REGION", "_Residence"], 
    var_name="Age_Gender", 
    value_name="COUNT"
)

# 分割 AgeGender 列為 AgeGroup 和 Gender
melted_df[["AGEGROUP", "GENDER"]] = melted_df["Age_Gender"].str.split("_", expand=True)

# 刪除原始的 AgeGender 列
melted_df = melted_df.drop(columns=["Age_Gender"])

# 刪除全部(ALL)
melted_df = melted_df[melted_df["_Residence"] != "全部(ALL)"]

# 確保數據格式正確並排序
melted_df["COUNT"] = pd.to_numeric(melted_df["COUNT"], errors="coerce")
melted_df = melted_df.dropna().sort_values(by=["_Years", "_Months", "REGION", "_Residence", "AGEGROUP", "GENDER"]).reset_index(drop=True)

# 分割國家中英文
melted_df[["RESIDENCE_CH", "RESIDENCE"]] = melted_df["_Residence"].str.split("(", expand=True)

# 處理英文國家欄位
melted_df["RESIDENCE"] = melted_df["RESIDENCE"].str.replace(")", "")

# 分隔地區中英文
melted_df[["REGION_CH", "REGION"]] = melted_df["REGION"].str.split("(", expand=True)
melted_df["REGION"] = melted_df["REGION"].str.replace(")", "")

# 增加欄位 YYYMM 日期欄位
melted_df = melted_df[melted_df["_Years"].apply(lambda x: str(x).isdigit())]  # 過濾非數字的行
melted_df["YYYY"] = melted_df["_Years"].astype(int) + 1911
melted_df["MM"] = melted_df["_Months"].str.zfill(2)
melted_df["YYYYMM"] = melted_df["YYYY"].astype(str) + melted_df["MM"]

# 重新排序欄位
final_df = melted_df[["_Years", "YYYY", "MM", "YYYYMM", "REGION_CH", "REGION", "RESIDENCE_CH", "RESIDENCE", "AGEGROUP", "GENDER", "COUNT"]].copy()

# 重新命名欄位
final_df.rename(columns={"_Years": "YYY"}, inplace=True)

# 儲存結果
final_df.to_csv("normalized.csv", index=False)