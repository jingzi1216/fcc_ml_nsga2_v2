import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination


# -----------------------------
# 1. 缓存加载模型
def load_model_rf():
    return joblib.load('rf_model.pkl')
def load_model_gb():
    return joblib.load('gb_model.pkl')


# 2. 定义界面布局
st.set_page_config(layout='wide')
st.title('催化裂化装置预测-优化模型')
st.write(
    '本模型针对催化裂化装置模拟数据，通过机器学习算法构建预测模型，并结合多目标优化算法，以原料质量流量等多维度特征为自变量，在满足汽油、液化气等产品性能指标约束条件下，对装置的经济价值（综合汽油、液化气和丙烯等产量及价值因素计算）和烟气中 CO₂ 排放量进行优化，输出综合最优解相关的自变量设定与预测目标值等信息，实现对催化裂化装置运行的智能优化模拟。')
if 'results' not in st.session_state:
    st.session_state.results = None

# 3. 配置优化参数
feature_bounds = {
    '原料质量流量t/h': (360, 420),
    '原料芳烃含量wt%': (15, 35),
    '原料镍含量ppmwt': (0.1, 2),
    '原料钒含量ppmwt': (0.1, 2),
    '原料残炭含量 wt%': (0.2, 0.6),
    '原料预热温度℃': (170, 220),
    '反应压力bar_g': (0.25, 4),
    '反应温度℃': (495, 525),
    '催化剂微反活性t%': (45, 55),
    '新鲜催化剂活性 wt%': (55, 70),
    '反应器密相催化剂藏量kg': (200000, 280000),
    '再生器床温℃': (660, 700),
    '原料比重g/cm3': (0.88, 0.92),
    '原料氮含量wt%': (0.01, 0.2),
    '原料硫含量wt%': (0.2, 0.5),
    '催化剂补充速率tonne/d': (2, 6),
    '提升蒸汽注入量tonne/hr': (4, 8),
    '雾化蒸汽注入量tonne/hr': (12, 18),
    '汽提蒸汽注入量tonne/hr': (4, 8)
}
feature_names = list(feature_bounds.keys())

target_columns = [
    '汽油收率wt%', '汽油芳烃含量vol %', '汽油烯烃含量vol%',
    '汽油RON', '汽油干点℃', '液化气收率wt%',
    '液化气丙烯含量wt%', '液化气C5体积比 vol%',
    '烟气中CO2排放量t/h', '柴油ASTM D8695% ℃'
]
target_names = target_columns.copy()
target_bounds = {
    '汽油收率wt%': (35, 55),
    '汽油芳烃含量vol %': (0, 33),
    '汽油烯烃含量vol%': (0, 25),
    '汽油RON': (92, float('inf')),
    '汽油干点℃': (0, 215),
    '液化气收率wt%': (15, 35),
    '液化气丙烯含量wt%': (30, float('inf')),
    '液化气C5体积比 vol%': (0, 2.3),
    '柴油ASTM D8695% ℃': (0, 360)
}
def round_repair_X(pop, **kwargs):
    X = np.round(pop.get('X'), 2)
    pop.set('X', X)
    return pop
# 3. 优化问题类
def your_value_co2_ratio(y_pred, feed_flow):
    gas_frac = y_pred[target_names.index('汽油收率wt%')] / 100.0
    lpg_frac = y_pred[target_names.index('液化气收率wt%')] / 100.0
    prop_frac = y_pred[target_names.index('液化气丙烯含量wt%')] / 100.0
    co2 = y_pred[target_names.index('烟气中CO2排放量t/h')]
    gas_prod = gas_frac * feed_flow
    lpg_prod = lpg_frac * feed_flow
    prop_prod = lpg_prod * prop_frac
    value = gas_prod * 1.2 + (lpg_prod - prop_prod) * 1.0 + prop_prod * 1.5
    return value, co2, value / co2
class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        return np.round(X, 2)

# NSGA-II 问题定义
# 7. 定义优化问题
class MyProblem(ElementwiseProblem):
    def __init__(self, rf_regressor,gb_regressor, feature_names, target_names, target_bounds):
        self.rf_regressor = rf_regressor
        self.feature_names = feature_names
        self.gb_regressor = gb_regressor
        self.target_names = target_names
        self.target_bounds = target_bounds
        super().__init__(n_var=len(feature_names),
                         n_obj=2,
                         n_constr=len(target_bounds) * 2,
                         xl=np.array([feature_bounds[name][0] for name in feature_names]),
                         xu=np.array([feature_bounds[name][1] for name in feature_names]))

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x, 2)  # 确保评估时也使用两位小数
        x_df = pd.DataFrame([x], columns=self.feature_names)
        # RF 对所有目标的初步预测
        rf_pred = self.rf_regressor.predict(x_df)[0]
        # GB 对指定两个目标的预测
        gb_pred = self.gb_regressor.predict(x_df)[0]
        y_pred = self.rf_regressor.predict(x_df)[0]
        # 合并预测结果
        y_pred = rf_pred.copy()
        y_pred[self.target_names.index('烟气中CO2排放量t/h')] = gb_pred[self.target_names.index('烟气中CO2排放量t/h')]
        y_pred[self.target_names.index('液化气丙烯含量wt%')] = gb_pred[self.target_names.index('液化气丙烯含量wt%')]

        # 提取原料质量流量
        feed_flow = x[self.feature_names.index('原料质量流量t/h')]

        # 解析产出
        gas_frac = y_pred[self.target_names.index('汽油收率wt%')] / 100.0
        lpg_frac = y_pred[self.target_names.index('液化气收率wt%')] / 100.0
        prop_frac = y_pred[self.target_names.index('液化气丙烯含量wt%')] / 100.0
        co2 = y_pred[self.target_names.index('烟气中CO2排放量t/h')]

        # 计算产量
        gas_prod = gas_frac * feed_flow
        lpg_prod = lpg_frac * feed_flow
        prop_prod = lpg_prod * prop_frac

        # 计算价值
        value = gas_prod * 1.2 + (lpg_prod - prop_prod) * 1.0 + prop_prod * 1.5

        # 目标：最大化价值（取负值），最小化 CO2
        out["F"] = [-value, co2]

        # 约束
        constr = []
        for name, (lo, hi) in self.target_bounds.items():
            idx = self.target_names.index(name)
            yv = y_pred[idx]
            if lo is not None:
                constr.append(lo - yv)  # g(x) <= 0
            if hi is not None:
                constr.append(yv - hi)  # g(x) <= 0
        out["G"] = constr


# 开始优化按钮
if st.button('开始优化'):
    with st.spinner('优化进行中，请稍候...'):
        rf = load_model_rf()
        gb = load_model_gb()
        problem = MyProblem(rf, gb,feature_names, target_names, target_bounds)
        algorithm = NSGA2(pop_size=20,
                          sampling=FloatRandomSampling(),
                          crossover=SBX(prob=0.9, eta=15),
                          mutation=PM(eta=20),
                          repair=RoundRepair(),
                          eliminate_duplicates=True)
        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8, cvtol=1e-6, ftol=1e-6,
            period=20, n_max_gen=100, n_max_evals=100000)
        res = minimize(problem, algorithm, termination,
                       seed=41, save_history=False, verbose=True)
        # 构造结果 DataFrame
        X, F = res.X, res.F
        values = -F[:, 0];
        co2s = F[:, 1];
        ratio = values / co2s
        rows = []
        for i, x in enumerate(X):
            row = {f: x[j] for j, f in enumerate(feature_names)}
            # 对当前解用两个模型各自做一次完整预测
            x_df = pd.DataFrame([x], columns=feature_names)
            y_rf = rf.predict(x_df)[0]
            y_gb = gb.predict(x_df)[0]
            # 合并预测：默认用 RF，GB 覆盖两个目标
            y_merged = y_rf.copy()
            idx_co2 = target_names.index('烟气中CO2排放量t/h')
            idx_prop = target_names.index('液化气丙烯含量wt%')
            y_merged[idx_co2] = y_gb[idx_co2]
            y_merged[idx_prop] = y_gb[idx_prop]
            idx_gas = target_names.index('汽油收率wt%')
            y_merged[idx_gas] *= 0.965
            # 重新计算
            feed = x[feature_names.index('原料质量流量t/h')]
            value, co2, ratios = your_value_co2_ratio(y_merged, feed)

            # 填入各个目标值
            for k, tn in enumerate(target_names):
                row[tn] = y_merged[k]

            # 再加上价值、CO2、最优值
            row.update({
                '目标产品价值': value,
                '烟气中CO2排放': co2,
                '最优值': ratios
            })
            rows.append(row)
        st.session_state.results = pd.DataFrame(rows)
        st.success('优化完成！')

# 展示结果表格 & 综合最优解
if st.session_state.results is not None:
    st.subheader('结果展示')
    st.dataframe(st.session_state.results)

    # 计算综合最优
    df = st.session_state.results
    best_idx = df['最优值'].idxmax()
    best = df.loc[best_idx]

    st.markdown('---')
    st.subheader('综合最优解展示')
    # 自变量表格
    feat_df = best[feature_names].to_frame('值').reset_index().rename(columns={'index': '自变量'})
    # 目标值表格
    targ_df = best[target_columns].to_frame('值').reset_index().rename(columns={'index': '目标变量'})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('###### 自变量设定')
        st.table(feat_df)
    with col2:
        st.markdown('###### 预测因变量')
        st.table(targ_df)

    # 核心指标展示
    col3, col4, col5 = st.columns(3)
    col3.metric('目标产品价值', f"{best['目标产品价值']:.2f}")
    col4.metric('烟气中CO2 排放量', f"{best['烟气中CO2排放']:.2f}")
    col5.metric('最优值', f"{best['最优值']:.4f}")
