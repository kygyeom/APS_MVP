import streamlit as st
import pandas as pd
import datetime
import sys
import numpy as np
import plotly.graph_objects as go

# 사용자 정의 simglucose 경로 설정
sys.path.insert(0, './simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

# 1. 데이터 로딩
# df = pd.read_csv("adolescent#001_100_500.csv")
name = 'adult#001_100_500'
df = pd.read_csv(f"{name}.csv")

df["Time"] = pd.to_datetime(df["Time"])

# 3시간 샘플링 시각화
mid_index = len(df) // 2
sample_df = df.iloc[mid_index - 30: mid_index + 30].reset_index(drop=True)

# 전체 시간 구간 시각화
st.title("🩺 인슐린 제어 시뮬레이터")

# 1-1. 전체 혈당 & CGM 그래프
st.subheader("📈 전체 혈당 & CGM")
fig_bg_cgm = go.Figure()
fig_bg_cgm.add_trace(go.Scatter(x=df["Time"], y=df["BG"], name="혈당", line=dict(color="blue")))
fig_bg_cgm.add_trace(go.Scatter(
    x=df["Time"],
    y=df["CGM"],
    name="CGM",
    line=dict(color="green", dash="dot")
))

fig_bg_cgm.update_layout(
    xaxis_title="시간",
    yaxis_title="혈당 (mg/dL)",
    height=400
)
st.plotly_chart(fig_bg_cgm, use_container_width=True)

# 1-2. 전체 인슐린 주입량 그래프
fig_insulin = go.Figure()
fig_insulin.add_trace(go.Scatter(x=df["Time"], y=df["insulin"], name="인슐린", line=dict(color="red")))
fig_insulin.update_layout(
    xaxis_title="시간",
    yaxis_title="인슐린 (U)",
    height=300
)
st.plotly_chart(fig_insulin, use_container_width=True)

fig_cho = go.Figure()
fig_cho.add_trace(go.Scatter(x=df["Time"], y=df["CHO"], name="CHO", line=dict(color="orange")))
fig_cho.update_layout(
    xaxis_title="시간",
    yaxis_title="CHO (g)",
    height=300
)
st.plotly_chart(fig_cho, use_container_width=True)

# 1. AI 초기 혈당값 추출

# AI 제어 = 기존 데이터 기반 (샘플링된 6시간)
start_ai = mid_index - 60
ai_df = df.iloc[start_ai: start_ai + 120].reset_index(drop=True)
bg_ai = ai_df["BG"].values
ins_ai = ai_df["insulin"].values
init_state = bg_ai[0] +20

# 2. 사용자 인슐린 제어
st.subheader("🎛 사용자 인슐린 제어 (6시간)")
dose1 = st.slider("0~2시간", 0.0, 0.5, 0.1, 0.01)
dose2 = st.slider("2~4시간", 0.0, 0.5, 0.1, 0.01)
dose3 = st.slider("4~6시간", 0.0, 0.5, 0.1, 0.01)
user_doses = [dose1]*40 + [dose2]*40 + [dose3]*40  # 총 120 step = 6시간

# 시뮬레이션 준비
sensor = CGMSensor.withName("Dexcom")
pump = InsulinPump.withName("Insulet")
name = 'adult#001_100_500'
base_name = name.split('_')[0]  # 'adult#001'
patient = T1DPatient.withName(base_name, init_state=init_state)

scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)

# 사용자 제어 시뮬레이션
env_user = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
env_user.reset()
bg_user, ins_user = [], []

for u in user_doses:
    obs, _, _, _ = env_user.step(Action(basal=0.0, bolus=u))
    bg_user.append(obs[0])
    ins_user.append(u)



# 3-1. 혈당 비교 시각화
st.subheader("📈 혈당 비교: AI vs 사용자")
fig_bg = go.Figure()
fig_bg.add_trace(go.Scatter(y=bg_ai, name="AI 혈당", line=dict(color="green")))
fig_bg.add_trace(go.Scatter(y=bg_user, name="사용자 혈당", line=dict(color="blue")))
fig_bg.update_layout(
    xaxis_title="Time Step (3min)",
    yaxis_title="혈당 (mg/dL)",
    height=400
)
st.plotly_chart(fig_bg, use_container_width=True)

# 3-2. 인슐린 비교 시각화
st.subheader("💉 인슐린 주입량 비교")
fig_insulin = go.Figure()
fig_insulin.add_trace(go.Scatter(y=ins_ai, name="AI 인슐린", line=dict(color="orange", dash="dash")))
fig_insulin.add_trace(go.Scatter(y=ins_user, name="사용자 인슐린", line=dict(color="red")))
fig_insulin.update_layout(
    xaxis_title="Time Step (3min)",
    yaxis_title="인슐린 (U)",
    height=300
)
st.plotly_chart(fig_insulin, use_container_width=True)


# 4. TIR(Time in Range) 계산 및 비교
def compute_tir(bg_series):
    in_range = np.logical_and(np.array(bg_series) >= 70, np.array(bg_series) <= 180)
    return 100 * np.sum(in_range) / len(bg_series)

tir_ai = compute_tir(bg_ai)
tir_user = compute_tir(bg_user)

st.subheader("📊 TIR (Time in Range: 70~180 mg/dL)")
st.write(f"✅ **AI TIR**: {tir_ai:.2f}%")
st.write(f"🧑‍⚕️ **사용자 TIR**: {tir_user:.2f}%")

import plotly.graph_objects as go

# 5. TIR 시각화
fig_tir = go.Figure()
fig_tir.add_trace(go.Bar(
    x=["AI", "사용자"],
    y=[tir_ai, tir_user],
    marker_color=["green", "blue"]
))

fig_tir.update_layout(
    title="TIR (Time in Range) 비교",
    yaxis_title="TIR (%)",
    xaxis_title="제어 주체",
    yaxis=dict(range=[0, 100]),
    height=400
)

st.plotly_chart(fig_tir, use_container_width=True)