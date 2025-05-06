import polars as pl
import plotly.express as px

df = pl.read_csv('samples/fold_results.csv')
alpha = 0.2
fname = f'maskingpct_vs_step_alpha_{alpha}'

df = df.with_columns(
  pl.col('maskingpct').ewm_mean(alpha=alpha).alias('maskingpct_smooth')
)

fig = px.scatter(
  x=df['step'],
  y=df['maskingpct_smooth'],
)
fig.update_traces(mode='lines+markers', name='Raw')
fig.update_layout(xaxis_title='step', yaxis_title='masking token percentage')
fig.show()
fig.write_image(f'figures/{fname}.png', width=700, height=600, scale=2)
