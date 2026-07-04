"""
TASK 3: Exploratory Data Analysis (EDA)
Dataset: cleaned_online_store_orders.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["figure.dpi"] = 110
COLOR = "#5b8c5a"
PALETTE = ["#5b8c5a", "#8fbc8f", "#c9a24b", "#4a7a96", "#a15c5c", "#7a5c96", "#d98a3d"]

df = pd.read_csv("data/cleaned_online_store_orders.csv", parse_dates=["Date"])

log_lines = []
def log(x=""):
    print(x)
    log_lines.append(str(x))

log("=" * 60)
log("TASK 3: EXPLORATORY DATA ANALYSIS (EDA)")
log("=" * 60)

log("\n Basic Statistics ")
log(df[["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]].describe().round(2).to_string())

log(f"\nTotal Revenue (all orders): ${df['TotalPrice'].sum():,.2f}")
delivered = df[df["OrderStatus"].isin(["Delivered", "Shipped"])]
log(f"Realized Revenue (Delivered + Shipped only): ${delivered['TotalPrice'].sum():,.2f}")
log(f"Total Orders: {len(df):,}")
log(f"Average Order Value: ${df['TotalPrice'].mean():,.2f}")
log(f"Unique Customers: {df['CustomerID'].nunique():,}")
log(f"Repeat Customers (>1 order): {(df['CustomerID'].value_counts() > 1).sum()}")

log("\n Revenue by Product ")
prod_rev = df.groupby("Product")["TotalPrice"].sum().sort_values(ascending=False)
log(prod_rev.round(2).to_string())

log("\n Orders by Status ")
status_counts = df["OrderStatus"].value_counts()
log(status_counts.to_string())
log(f"Cancellation + Return rate: "
    f"{(df['OrderStatus'].isin(['Cancelled','Returned']).mean())*100:.1f}%")

log("\n Revenue by Payment Method ")
pay_rev = df.groupby("PaymentMethod")["TotalPrice"].sum().sort_values(ascending=False)
log(pay_rev.round(2).to_string())

log("\n Orders by Referral Source ")
ref_counts = df["ReferralSource"].value_counts()
log(ref_counts.to_string())

log("\n Coupon Usage ")
coupon_counts = df["CouponCode"].value_counts()
log(coupon_counts.to_string())
log(f"Average order value WITH coupon: ${df[df['UsedCoupon']]['TotalPrice'].mean():,.2f}")
log(f"Average order value WITHOUT coupon: ${df[~df['UsedCoupon']]['TotalPrice'].mean():,.2f}")

log("\n Correlation (numeric features) ")
corr = df[["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]].corr()
log(corr.round(2).to_string())

log("\n Monthly Order Volume & Revenue Trend ")
monthly = df.groupby("OrderMonth").agg(Orders=("OrderID", "count"), Revenue=("TotalPrice", "sum"))
log(monthly.round(2).to_string())

log("\n Outlier Check (IQR method on TotalPrice) ")
q1, q3 = df["TotalPrice"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["TotalPrice"] < lower) | (df["TotalPrice"] > upper)]
log(f"IQR bounds: [{lower:.2f}, {upper:.2f}]")
log(f"Outlier orders detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}% of orders)")

log("\n Key Findings ")
top_prod = prod_rev.index[0]
top_pay = pay_rev.index[0]
top_ref = ref_counts.index[0]
cancel_rate = df['OrderStatus'].isin(['Cancelled','Returned']).mean()*100
log(f"  1. '{top_prod}' generates the most revenue (${prod_rev.iloc[0]:,.2f}), reflecting its high unit price.")
log(f"  2. '{top_pay}' is the leading payment method by revenue (${pay_rev.iloc[0]:,.2f}).")
log(f"  3. '{top_ref}' is the top referral/acquisition channel ({ref_counts.iloc[0]} orders).")
log(f"  4. {cancel_rate:.1f}% of orders end up Cancelled or Returned - a meaningful share worth investigating operationally.")
log(f"  5. Orders using a coupon average ${df[df['UsedCoupon']]['TotalPrice'].mean():,.2f} vs "
    f"${df[~df['UsedCoupon']]['TotalPrice'].mean():,.2f} without one, "
    f"{'higher' if df[df['UsedCoupon']]['TotalPrice'].mean() > df[~df['UsedCoupon']]['TotalPrice'].mean() else 'lower'} on average.")
log(f"  6. TotalPrice correlates strongly with UnitPrice (r={corr.loc['UnitPrice','TotalPrice']:.2f}) and "
    f"moderately with Quantity (r={corr.loc['Quantity','TotalPrice']:.2f}); ItemsInCart shows weak correlation "
    f"(r={corr.loc['ItemsInCart','TotalPrice']:.2f}), suggesting cart size doesn't strongly predict order value.")
log(f"  7. {len(outliers)} orders ({len(outliers)/len(df)*100:.1f}%) are statistical high-value outliers by the IQR method.")

with open("reports/task3_eda_summary.txt", "w") as f:
    f.write("\n".join(log_lines))

#  Charts 

# 1. Revenue by Product
fig, ax = plt.subplots(figsize=(7, 4.5))
prod_rev.sort_values().plot(kind="barh", ax=ax, color=COLOR)
ax.set_title("Total Revenue by Product")
ax.set_xlabel("Revenue ($)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("01_revenue_by_product.png")
plt.close()

# 2. Monthly Revenue Trend
fig, ax = plt.subplots(figsize=(9, 4.5))
monthly["Revenue"].plot(kind="line", marker="o", ax=ax, color=COLOR, markersize=4)
ax.set_title("Monthly Revenue Trend (2023-2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
ax.grid(alpha=0.3)
step = max(1, len(monthly)//12)
ax.set_xticks(range(0, len(monthly), step))
ax.set_xticklabels(monthly.index[::step], rotation=45, ha="right")
plt.tight_layout()
plt.savefig("02_monthly_revenue_trend.png")
plt.close()

# 3. Order Status distribution (pie)
fig, ax = plt.subplots(figsize=(6, 6))
status_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=PALETTE, ylabel="")
ax.set_title("Orders by Status")
plt.tight_layout()
plt.savefig("03_order_status_share.png")
plt.close()

# 4. Order Value distribution (histogram)
fig, ax = plt.subplots(figsize=(7, 4.5))
df["TotalPrice"].plot(kind="hist", bins=30, ax=ax, color=COLOR, edgecolor="white")
ax.set_title("Distribution of Order Value")
ax.set_xlabel("Total Price ($)")
plt.tight_layout()
plt.savefig("04_order_value_distribution.png")
plt.close()

# 5. Revenue by Payment Method x Product (stacked)
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df.pivot_table(index="PaymentMethod", columns="Product", values="TotalPrice", aggfunc="sum")
pivot.plot(kind="bar", stacked=True, ax=ax, color=PALETTE)
ax.set_title("Revenue by Payment Method, Split by Product")
ax.set_ylabel("Revenue ($)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("charts/05_payment_product_breakdown.png")
plt.close()

# 6. Referral source vs order count
fig, ax = plt.subplots(figsize=(7, 4.5))
ref_counts.sort_values().plot(kind="barh", ax=ax, color=COLOR)
ax.set_title("Orders by Referral Source")
ax.set_xlabel("Number of Orders")
plt.tight_layout()
plt.savefig("06_referral_source.png")
plt.close()

print("\nSaved 6 charts to charts/")
print("Saved -> reports/task3_eda_summary.txt")
