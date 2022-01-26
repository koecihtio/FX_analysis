# 開発マニュアル
## モジュール
### make_dataset
価格データをロードして、分析に用いるデータセットを構築するためのモジュール。使用する通貨ペアや指数を選択すると、自動でデータセットが構築されるように設計する。

### prediction
make_datasetモジュールを用いて構築されたデータセットに対して、価格の予測値を出力するためのモジュール。説明変数や学習器を選択できるように設計する。