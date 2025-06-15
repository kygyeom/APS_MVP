import streamlit as st
import pandas as pd
import datetime
import sys
import plotly.graph_objects as go
import copy

# 사용자 정의 simglucose 경로 설정
sys.path.insert(0, './simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

def show_section_info(df, env, section_index):
    STEP_PER_SECTION = 160  # 3분 간격 × 160 = 480분 = 8시간
    start = section_index * STEP_PER_SECTION
    end = start + STEP_PER_SECTION

    # ⏱ 정확한 시간 샘플링 (df 기반)
    section_start_time = pd.to_datetime(df["Time"].iloc[start])
    section_end_time = pd.to_datetime(df["Time"].iloc[end - 1])

    # 🩸 현재 혈당
    try:
        if hasattr(env, "history") and len(env.history) > 0:
            last_obs = env.history[-1]
            if hasattr(last_obs, "observation"):
                current_bg = last_obs.observation[0]
            elif isinstance(last_obs, dict) and "observation" in last_obs:
                current_bg = last_obs["observation"][0]
            else:
                current_bg = df["BG"].iloc[start]
        else:
            current_bg = df["BG"].iloc[start]
    except Exception:
        current_bg = df["BG"].iloc[start]
    current_bg = round(current_bg, 1)

    # 🍽 식사 정보
    meal_df = df.iloc[start:end][df["CHO"] > 0]
    if not meal_df.empty:
        meal_events = []
        for _, row in meal_df.iterrows():
            time = pd.to_datetime(row["Time"]).strftime("%H:%M")
            cho = round(row["CHO"], 1)
            meal_events.append(f"{time} - {cho}g")
        meal_info = "🍽 식사 시점 및 섭취량:\n- " + "\n- ".join(meal_events)
    else:
        meal_info = "🥛 공복 상태: 해당 구간에 식사 없음"

    # 🏃 활동 정보
    section_map = {
        0: "🌅 수면 시간",
        1: "🏃‍♂️ 일반 활동",
        2: "🌇 저녁 전 안정기"
    }
    activity = section_map.get(section_index, "📌 일반 구간")

    # 💉 권장 인슐린 계산
    target_bg = 110
    gf = 50
    icr = 10
    meal_carb = df.iloc[start:end]["CHO"].sum()
    correction = max((current_bg - target_bg), 0) / gf
    meal_insulin = meal_carb / icr / 3
    recommended_bolus = round(correction + meal_insulin / 3, 2)

    col1, col2 = st.columns([1, 2])  # 비율 조정 가능

    with col1:
        st.image("CGM.png", caption="혈당 측정기", use_container_width=True)

    with col2:
        # st.subheader(f"🔴 구간 {section_index + 1}")
        st.markdown(f"⏱ **시간**: {section_start_time.strftime('%H:%M')} ~ {section_end_time.strftime('%H:%M')}")
        st.markdown(f"🩸 **현재 혈당**: `{current_bg} mg/dL`")
        st.markdown(meal_info)
        st.markdown(f"📌 **활동 정보**: {activity}")

    st.info(f"""
    ### 💉 권장 인슐린 계산 정보
    - 목표 혈당: {target_bg} mg/dL
    - 감도 계수(GF): {gf}, ICR: {icr}
    - 예상 식사량: **{meal_carb:.1f} g 탄수화물**
    ➡️ 권장 볼루스 인슐린: **{recommended_bolus} 단위**
    """)

    with st.expander("📘 인슐린 주입 기준 보기", expanded=False):
        st.markdown("""
        - **볼루스 인슐린**: 식사량에 따라 설정합니다 (10g CHO 당 1.0 단위)
        - **기저 인슐린**: 식사와 관계없이 지속적으로 작용합니다 (보통 0.01~0.03 단위/step)
        - 총 주입량은 `단위/step × 160 step / 3 (8시간)`으로 계산됩니다
        """)




# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

st.title("🩺 인슐린 제어 시뮬레이터")

# STEP 0: 환자 선택
if st.session_state.step == 0:
    st.subheader("1️⃣ 환자 선택")
    patient_name = st.selectbox("시뮬레이션할 환자를 선택하세요:", [
        "adult#001", "adult#002", "adult#003",
        "adolescent#001", "adolescent#002", "adolescent#003",
    ])
    if st.button("다음 단계로"):
        st.session_state.selected_patient = patient_name
        st.session_state.csv_file = f"{patient_name}_100_500.csv"
        st.session_state.step = 1
        st.rerun()

# STEP 1: 데이터 불러오기
elif st.session_state.step == 1:
    df = pd.read_csv(f"data/{st.session_state.csv_file}")
    df["Time"] = pd.to_datetime(df["Time"])
    st.session_df = df
    st.write(f"✅ {st.session_state.selected_patient} 데이터 로딩 완료")
    if st.button("➡️ 시뮬레이션 시작"):
        st.session_state.step = 21
        st.rerun()

# STEP 21~23: 각 구간별 시뮬레이션
for seg in [1, 2, 3]:
    if st.session_state.step == 20 + seg:
        df = st.session_df
        ai_df = df.iloc[:480].reset_index(drop=True)
        init_bg = ai_df["BG"].iloc[0]

        if seg == 1 and "env_user" not in st.session_state:
            sensor = CGMSensor.withName("Dexcom")
            pump = InsulinPump.withName("Insulet")
            patient = T1DPatient.withName(st.session_state.selected_patient, init_state=init_bg)
            scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)
            env_user = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
            env_user.reset()
            st.session_state.env_user = env_user

        env = st.session_state.env_user
        st.subheader(f"🧪 구간 {seg}: {(seg - 1) * 6}~{seg * 6} 시간")

        dose_key = f"dose{seg}"
        bg_key = f"bg_user{seg}"
        env_init_key = f"env_state_{seg}"
        env_result_key = f"env_result_{seg}"

        # 예: STEP 21, 22, 23에서 각 구간 정보 보여줄 때
        section_index = st.session_state.step - 21
        show_section_info(df, env, section_index)
        
        dose = st.slider(f"구간 {seg} 볼루스 인슐린 (단위)", 0.0, 0.2, 0.05, 0.005, key=dose_key)
        basal = st.slider("기저 인슐린 (전 구간 적용)", 0.0, 0.05, st.session_state.get("dose_basal", 0.02), 0.001, key=f"basal{seg}")

        # 💉 총 인슐린 투여량 계산
        total_basal = round(basal * 160, 2)  # 160 스텝 동안의 총 기저 인슐린
        total_bolus = dose                   # 한 번에 주입
        total_insulin = round(total_basal + total_bolus, 2)

        # 💬 사용자에게 보여주기
        st.markdown(f"""
        🔢 **8시간 총 인슐린 투여량**  
        - 💧 기저 인슐린: `{basal} × 160 = {total_basal} 단위`  
        - 💉 볼루스 인슐린: `{total_bolus} 단위`  
        - ✅ **총 투여량**: `{total_insulin} 단위`
        """)

        if env_init_key not in st.session_state:
            st.session_state[env_init_key] = copy.deepcopy(env)

        if st.button(f"▶ 구간 {seg} 실행"):
            env = copy.deepcopy(st.session_state[env_init_key])
            result = []
            for _ in range(160):
                obs, _, _, _ = env.step(Action(basal=basal, bolus=dose))
                result.append(obs[0])
            st.session_state[bg_key] = result
            st.session_state[env_result_key] = copy.deepcopy(env)

            # ⏱ x축: 시작 시간 + 3분 간격 × 스텝
            start_time = datetime.datetime.strptime("00:00", "%H:%M") + datetime.timedelta(minutes=(seg - 1) * 160 * 3)
            time_range = [start_time + datetime.timedelta(minutes=3 * i) for i in range(160)]

            # 📈 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_range,
                y=result,
                mode="lines",
                name=f"구간 {seg} 결과",
                line=dict(color="red")
            ))
            fig.update_layout(
                title=f"구간 {seg} 혈당 반응",
                xaxis_title="시간 (시:분)",
                yaxis_title="혈당 (mg/dL)",
                xaxis=dict(
                    tickformat="%H:%M",
                    tickangle=45
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
        if st.button(f"🔁 구간 {seg} 다시 설정"):
            if bg_key in st.session_state:
                del st.session_state[bg_key]

        if st.button("➡️ 다음 구간으로"):
            st.session_state.env_user = copy.deepcopy(st.session_state[env_result_key])
            st.session_state.dose_basal = basal
            st.session_state.step += 1
            st.rerun()

# STEP 24: 결과 통합 시각화
if st.session_state.step == 24:
    st.subheader("✅ 전체 시뮬레이션 결과 요약")
    fig = go.Figure()
    full_bg = []
    full_bolus = []
    full_basal = []

    for i in range(1, 4):
        bg_key = f"bg_user{i}"
        dose_key = f"dose{i}"
        if bg_key in st.session_state:
            full_bg.extend(st.session_state[bg_key])
            dose_value = st.session_state.get(dose_key, 0.0)
            full_bolus.extend([dose_value] * 160)
            full_basal.extend([st.session_state.get("dose_basal", 0.02)] * 160)

    # AI 제어 결과 불러오기
    ai_df = st.session_df.iloc[:480].reset_index(drop=True)
    ai_bg = ai_df["BG"].tolist()

    # 혈당 비교 시각화
    fig = go.Figure()

    # 사용자 혈당
    fig.add_trace(go.Scatter(
        y=full_bg,
        mode="lines",
        name="사용자 제어 혈당",
        line=dict(color="blue")
    ))

    # AI 혈당
    fig.add_trace(go.Scatter(
        y=ai_bg,
        mode="lines",
        name="AI 제어 혈당",
        line=dict(color="gray", dash="dot")
    ))

    fig.update_layout(
        title="AI vs 사용자 제어 시뮬레이션 결과 (14시간)",
        xaxis_title="Time Step (5 min)",
        yaxis_title="값 (mg/dL 또는 단위)",
        legend=dict(x=0, y=1.1, orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("✅ 시뮬레이션 완료 → 처음으로"):
        for key in list(st.session_state.keys()):
            if key.startswith("bg_user") or key.startswith("env_") or key.startswith("dose"):
                del st.session_state[key]
        st.session_state.step = 0
        st.rerun()