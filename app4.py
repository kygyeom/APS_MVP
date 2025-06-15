import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
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

components.html("""
    <script>
        window.scrollTo({ top: 0, behavior: 'smooth' });
    </script>
""", height=0)

    
if st.session_state.get("trigger_scroll", False):
    components.html("""
        <script>
            window.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
    """, height=0)
    st.session_state.trigger_scroll = False  # 플래그 초기화
    
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

    # 🍽 식사 정보 + 음식 예시 추가 (다양한 종류로 확장)
    meal_df = df.iloc[start:end][df["CHO"] > 0]
    if not meal_df.empty:
        meal_events = []
        for _, row in meal_df.iterrows():
            time = pd.to_datetime(row["Time"]).strftime("%H:%M")
            cho = round(row["CHO"], 1)
            
            # 탄수화물 → 음식 예시 및 칼로리 대략 추정
            if cho < 10:
                food = "딸기 한 줌 🍓 / 우유 1컵 🥛"
            elif cho < 20:
                food = "바나나 1개 🍌 / 고구마 반 개 🍠"
            elif cho < 30:
                food = "식빵 1.5장 🍞 / 그래놀라 요거트 🥣"
            elif cho < 40:
                food = "공기밥 반 공기 🍚 / 토스트 세트 🍳"
            elif cho < 55:
                food = "라면 1개 🍜 / 김밥 1줄 🍙"
            elif cho < 70:
                food = "한식 도시락 🍱 / 떡볶이 + 순대 🍢"
            else:
                food = "햄버거 세트 🍔🍟 / 피자 2조각 🍕"

            estimated_kcal = int(cho * 4)  # 탄수화물 1g = 약 4 kcal

            meal_events.append(
                f"{time}쯤에 {food}를 먹었어요. "
                f"탄수화물 약 {cho}g → 약 {estimated_kcal} kcal 정도 됩니다."
            )

        meal_info = "🍽 식사 기록 요약:\n- " + "\n- ".join(meal_events)
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
    meal_insulin = meal_carb / icr
    recommended_bolus = round(correction + meal_insulin, 2)

    # 볼루스 주입 시점 계산 (식사 30분 전, CHO >= 30g인 경우만)
    bolus_time_info = ""
    main_meals = df.iloc[start:end][df["CHO"] >= 10]
    if not main_meals.empty:
        first_meal_time = pd.to_datetime(main_meals["Time"].iloc[0])
        bolus_time = first_meal_time - pd.Timedelta(minutes=30)
        df_section = df.iloc[start:end].reset_index(drop=True)
        df_section["Time"] = pd.to_datetime(df_section["Time"])
        bolus_idx = (df_section["Time"] - bolus_time).abs().idxmin()
        bolus_time_info = f"🍚 주요 식사 감지됨: {first_meal_time.strftime('%H:%M')}\n💉 볼루스 인슐린은 `{bolus_time.strftime('%H:%M')}`에 1회 주입 예정 (step {bolus_idx})"

    st.markdown(f"⏱ **시간**: {section_start_time.strftime('%H:%M')} ~ {section_end_time.strftime('%H:%M')}, {activity}")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("CGM.png", caption="혈당 측정기", use_container_width=True)

    with col2:

        st.markdown(f"🩸 **측정된 현재 혈당**: `{current_bg} mg/dL`, 권장 범위: 70~180 mg/dL")
        st.markdown(meal_info)
        if bolus_time_info:
            st.success(bolus_time_info)

    st.info(f"""
    ### 💉 권장 인슐린 계산 정보
    - 목표 혈당: {target_bg} mg/dL
    - 감도 계수(GF): {gf}, ICR: {icr}
    - 예상 식사량: **{meal_carb:.1f} g 탄수화물**
    ➡️ 권장 볼루스 인슐린: **{recommended_bolus} 단위**
    """)

    with st.expander("📘 인슐린 주입 기준 보기", expanded=False):
        st.markdown("""
        #### 💉 인슐린 주입 기준 안내

        - **볼루스 인슐린**: 식사량에 따라 설정하며, 주요 식사 전 **30분에 1회** 주입합니다.
            - 탄수화물 섭취량에 따른 계산 공식:
                ```
                볼루스 인슐린 (단위) = 식사 탄수화물(g) / ICR + 보정량
                ```
                - 예: 탄수화물 60g 섭취, ICR=10 → `60 / 10 = 6 단위`
                - 보정량 = (현재 혈당 - 목표 혈당) / 감도 계수(GF)

        - **기저 인슐린**: 식사와 관계없이 지속적으로 작용합니다.  
        보통 1 step (3분)마다 `0.01 ~ 0.03 단위`가 투여됩니다.

            - 총 기저 인슐린 양 계산 공식:
                ```
                기저 인슐린 총량 = 단위/step × 160 step (8시간)
                ```

            - 예: 0.02 단위/step이면 → `0.02 × 160 = 3.2 단위`

        ---
        ⚠️ 참고: 인슐린 용량은 개인의 인슐린 감수성에 따라 달라질 수 있으며,  
        본 시뮬레이터는 교육용으로 제공됩니다.
        """)

def analyze_glucose_events(bg_series, time_series):
    df_g = pd.DataFrame({"time": time_series, "bg": bg_series})
    df_g["status"] = "정상"
    df_g.loc[df_g["bg"] < 70, "status"] = "저혈당"
    df_g.loc[df_g["bg"] > 180, "status"] = "고혈당"

    df_g["group"] = (df_g["status"] != df_g["status"].shift()).cumsum()
    events = df_g[df_g["status"] != "정상"].groupby("group")

    messages = []
    for _, group in events:
        status = group["status"].iloc[0]
        t_start = group["time"].iloc[0].strftime("%H:%M")
        t_end = group["time"].iloc[-1].strftime("%H:%M")
        messages.append(f"- **{t_start} ~ {t_end}** 사이에 **{status}** 발생")

    return messages, df_g

def summarize_today(basal_list, bolus_list, meal_total, bg_series):
    summary = []

    # 총량 계산
    basal_total = sum(basal_list)
    bolus_total = sum(bolus_list)
    insulin_total = basal_total + bolus_total

    # # 1. 인슐린 총량 평가
    # if insulin_total > 10:
    #     summary.append("💉 인슐린을 전반적으로 많이 사용했습니다.")
    # elif insulin_total < 5:
    #     summary.append("💉 인슐린 사용량이 다소 부족했습니다.")
    # else:
    #     summary.append("💉 인슐린 용량은 적절한 수준이었습니다.")

    # 2. 식사량 평가
    if meal_total > 150:
        summary.append("🍚 오늘 섭취한 탄수화물 양이 많아 혈당 조절이 어려웠을 수 있습니다.")
    elif meal_total < 50:
        summary.append("🥛 식사량이 적어 저혈당 위험이 있을 수 있습니다.")
    else:
        summary.append("🥗 적절한 식사량이 유지되었습니다.")

    # 3. 혈당 패턴 평가
    hypo = sum(bg < 70 for bg in bg_series)
    hyper = sum(bg > 180 for bg in bg_series)

    if hypo > 5:
        summary.append("⚠️ 저혈당이 여러 차례 발생했습니다. 기저 인슐린을 줄이는 것이 좋겠습니다.")
    elif hyper > 5:
        summary.append("⚠️ 고혈당이 자주 발생했습니다. 식사량을 조절하거나 볼루스 인슐린을 늘려야 할 수 있습니다.")
    else:
        summary.append("✅ 혈당이 안정적으로 유지되었습니다.")

    # 종합 제안
    if hyper > 5 and meal_total > 150:
        summary.append("📌 식사량을 줄이거나 식후 가벼운 운동을 병행하면 혈당 조절에 도움이 됩니다.")
    elif hypo > 5 and insulin_total > 10:
        summary.append("📌 인슐린 용량을 줄이고 간식을 적절히 배분하는 것이 필요합니다.")
    elif 0 < hypo <= 5 or 0 < hyper <= 5:
        summary.append("📌 혈당 조절이 거의 잘 되었으나 약간의 보완 여지가 있습니다.")

    return "\n".join(summary)


# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

st.title("🩺 인슐린 제어 시뮬레이터")

# STEP 0: 환자 선택
if st.session_state.step == 0:

    st.markdown("""
    ### 👨‍⚕️ 인슐린 제어 시뮬레이터 소개

    가상의 제1형 당뇨병 환자를 대상으로,  
    **사용자가 직접 인슐린 주입량을 설정**하고  
    **AI 제어와 비교**해볼 수 있는 학습형 시뮬레이터입니다.

    ---

    #### 🎯 체험 목적 요약
    - 당뇨병 환자의 혈당 조절 어려움 **간접 체험**
    - 인슐린 **타이밍과 용량**의 중요성 학습
    - **기저 인슐린과 볼루스 인슐린**의 역할 이해
    """)

    st.markdown("## 💡 왜 혈당 조절이 어려울까요?")

    st.markdown("""
    당뇨병 환자에게 인슐린 조절은 매일 반복되는 과제입니다.  
    그중에서도 **식사와 혈당 측정**은 생명과 직결된 요소입니다.
    """)
    # ⏱ 혈당 조절 관련 팩트 카드
    with st.expander("🩸 혈당 스파이크란?"):
        st.markdown("""
        #### 📊 혈당 조절, 왜 중요할까요?

        - 🍚 건강한 사람은 식사 후 혈당이 **140mg/dL 이하**로 조절됩니다.  
        - 그러나 당뇨 환자는 쉽게 **180mg/dL 이상**으로 올라가며 이를 **혈당 스파이크**라고 부릅니다.
        - 💥 이 상태가 반복되면 **신장, 신경, 혈관계에 심각한 손상**을 줄 수 있습니다.
        - ⏱ 인슐린 주입이 30분만 늦어져도 혈당 조절은 큰 영향을 받습니다.

        ---
        🤖 AI는 정확한 시점과 용량을 계산해 최적의 인슐린 제어를 시도합니다.  
        🧑 사용자도 이를 직접 조절해보며 **혈당 반응을 체험**할 수 있습니다.
        """)

    # 💉 인슐린 타입 설명
    with st.expander("💉 인슐린 이란?"):
        st.markdown("""
        인슐린은 몸 안에서 혈당을 조절해주는 생명에 꼭 필요한 호르몬입니다.
        당뇨병 환자는 이 인슐린을 제대로 만들거나 활용하지 못해,
        식사 후 혈당이 급격히 올라가고 몸에 큰 부담을 주게 됩니다.

        특히 제1형 당뇨병 환자는 몸속에서 인슐린을 전혀 만들지 못하기 때문에,
        하루에도 여러 번 주사나 인슐린 펌프를 통해 외부에서 직접 주입해야 합니다.

        🩸 이 주사는 단순히 불편한 것을 넘어
        매일 반복되는 고통과 스트레스를 동반합니다.
        "지금 얼마나 넣어야 할까?", "혹시 저혈당이 올까?"라는 불안감은
        환자들의 일상에 늘 그림자처럼 따라붙습니다.

        이 시뮬레이터는
        그들의 하루를 조금이나마 체험해보고,
        AI의 도움으로 어떻게 부담을 줄일 수 있을지 함께 고민해보기 위해 만들어졌습니다.
                    
        | 구분 | 기저 인슐린 (Basal) | 식전 볼루스 인슐린 (Bolus) |
        |------|--------------------|-----------------------------|
        | 역할 | 공복 혈당 조절     | 식후 혈당 급등 억제         |
        | 타이밍 | 하루 1~2회 지속 주입 | 식사 30분 전               |
        | 작용 시간 | 느리고 지속적     | 빠르고 단기적              |

        """, unsafe_allow_html=True)

    with st.expander("🍽 식사는 왜 신중해야 하나요?"):
        st.markdown("""
        - **먹는 음식이 곧 혈당**입니다.  
        - 같은 음식도 **시간, 양, 활동량**에 따라 혈당 반응이 달라집니다.
        - 식사 전 인슐린(볼루스)을 **적절한 양으로, 미리** 주입하지 않으면  
        → 혈당이 **180mg/dL 이상으로 급등(스파이크)**할 수 있습니다.
        """)

    with st.expander("🩸 혈당 측정 어떻게 하나요?"):
        st.markdown("""
        - 당뇨병 환자는 하루에도 **여러 번 혈당을 측정**합니다.
        - 이는 단순한 숫자가 아니라,  
        **“지금 내 몸은 안전한가?”를 확인하는 생존의 도구**입니다.
        - 측정 없이 인슐린을 맞으면 → **저혈당 쇼크**나 **과다투여 위험** 발생
        """)

    # 📈 TIR 설명
    with st.expander("📈 TIR(Time in Range)이란?"):
        st.markdown("""
        - 혈당이 **70~180 mg/dL** 범위 내에 머무는 시간 비율입니다.  
        - TIR이 높을수록 혈당이 안정적으로 유지되고 있다고 볼 수 있습니다.  
        - 본 시뮬레이터에서는 **AI vs 사용자** TIR을 비교하여 인슐린 전략의 효과를 확인할 수 있습니다.
        """)

    st.markdown("---")
    st.success("✅ 아래에서 시뮬레이션할 환자를 선택한 후, 다음 단계로 이동하세요.")

    st.subheader("환자 선택")
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

        # CSV 불러오기 (앱 시작 시 한 번만 실행되게 outside에 둘 수도 있음)
    df_params = pd.read_csv("vpatient_params.csv")  # 경로에 맞게 수정

    # 선택된 환자 이름
    patient_name = st.session_state.selected_patient

    # 해당 환자 데이터 필터링
    patient_info = df_params[df_params["Name"] == patient_name]


        # 정보가 존재할 경우 출력
    if not patient_info.empty:
        info = patient_info.iloc[0]

        st.subheader(f"🧬 `{patient_name}` 환자 요약 정보")

        # 그룹 분류
        if "adolescent" in patient_name:
            group = "청소년"
        elif "adult" in patient_name:
            group = "성인"
        else:
            group = "기타"

        # 주요 생리학적 정보 출력
        st.markdown(f"""
        - **환자 그룹**: {group}  
        - **몸무게 (BW)**: {info['BW']:.3f} kg  
        - **인슐린 감수성 (u2ss)**: {info['u2ss']:.3f}  
        - **간 포도당 생성률 (kp1)**: {info['kp1']:.2f}  
        - **속효성 인슐린 흡수 속도 (ka1)**: {info['ka1']:.4f}  
        - **복부 피하 인슐린 반응 (isc1ss)**: {info['isc1ss']:.2f}  
        """)
    else:
        st.warning("선택한 환자의 정보를 찾을 수 없습니다.")

    with st.expander("ℹ️ 혈당 조절 가이드", expanded=False):
        st.markdown("""
        ### 🔄 혈당 조절 과정 안내

        1. **현재 혈당 확인**  
        - CGM(연속혈당측정기)을 통해 실시간 혈당을 확인합니다.

        2. **식사량 및 시점 파악**  
        - 해당 구간 내 식사 여부, 탄수화물 섭취량(CHO)을 확인합니다.

        3. **인슐린 계산 (추천값 제공)**  
        - 목표 혈당: `110 mg/dL`  
        - 감도 계수(GF): `50`  
        - 탄수화물 인슐린 비율(ICR): `10g 당 1.0 단위`  
        ➡️ 계산된 **추천 볼루스 인슐린**을 제공합니다.

        4. **기저 & 볼루스 인슐린 주입**  
        - **기저 인슐린**: 8시간 동안 일정하게 분산 주입 (예: 0.02 단위/step)  
        - **볼루스 인슐린**: 식사 30분 전에 한 번에 주입

        5. **시뮬레이션 실행**  
        - 설정한 인슐린 주입량에 따라 8시간 혈당 반응을 시뮬레이션합니다.

        6. **결과 분석 및 피드백**  
        - 최종 혈당이 정상 범위(70~180 mg/dL)인지 확인  
        - 고혈당/저혈당 발생 시간 요약  
        - 인슐린 용량 조절에 대한 피드백을 제공합니다.
            """)

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
        # st.subheader(f"🧪 구간 {seg}: {(seg - 1) * 6}~{seg * 6} 시간")

        dose_key = f"dose{seg}"
        bg_key = f"bg_user{seg}"
        env_init_key = f"env_state_{seg}"
        env_result_key = f"env_result_{seg}"

        # 예: STEP 21, 22, 23에서 각 구간 정보 보여줄 때
        section_index = st.session_state.step - 21
        show_section_info(df, env, section_index)
        
        dose = st.slider(f"볼루스 인슐린 (식사 30분전 주입)", 0.0, 5.0, 1.0, 0.1, key=dose_key)
        basal = st.slider("기저 인슐린 (8시간 동안 주입)", 0.0, 0.05, st.session_state.get("dose_basal", 0.02), 0.001, key=f"basal{seg}")

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

        if st.button(f"시뮬레이션 {seg} 실행"):
            env = copy.deepcopy(st.session_state[env_init_key])
            result = []

            # 1️⃣ 식사 시점 탐지 및 볼루스 주입 시점 설정
            section_df = df.iloc[seg * 160 : (seg + 1) * 160]
            meal_times = section_df[section_df["CHO"] >= 30].index.tolist()
            bolus_step = None
            if meal_times:
                meal_step = meal_times[0] - section_df.index[0]  # 상대적 위치
                bolus_step = max(meal_step - 10, 0)  # 30분 전

            for t in range(160):
                bolus = dose if bolus_step == t else 0.0
                obs, _, _, _ = env.step(Action(basal=basal, bolus=bolus))
                result.append(obs[0])
            st.session_state[bg_key] = result
            st.session_state[env_result_key] = copy.deepcopy(env)

            # ⏱ x축: 시작 시간 + 3분 간격 × 스텝
            start_time = datetime.datetime.strptime("00:00", "%H:%M") + datetime.timedelta(minutes=seg * 160 * 3)
            time_range = [start_time + datetime.timedelta(minutes=3 * i) for i in range(160)]

            # 🥗 현재 구간에 해당하는 식사량 시계열
            section_df = df.iloc[seg * 160 : (seg + 1) * 160].reset_index(drop=True)
            meal_series = section_df["CHO"].tolist()

            # 📈 복합 시각화
            fig = go.Figure()

            # ✅ 1. 정상 범위 음영 (70~180 mg/dL)
            fig.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=time_range[0], x1=time_range[-1],
                y0=70, y1=180,
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0,
            )


            # 1️⃣ 혈당 선 그래프
            fig.add_trace(go.Scatter(
                x=time_range,
                y=result,
                mode="lines",
                name="혈당 (mg/dL)",
                line=dict(color="red"),
                yaxis="y1"
            ))

            # 2️⃣ 식사량 막대 그래프
            fig.add_trace(go.Bar(
                x=time_range,
                y=meal_series,
                name="식사량 (CHO g)",
                marker_color="lightblue",
                opacity=0.6,
                yaxis="y2"
            ))

            # 📐 그래프 레이아웃
            fig.update_layout(
                title=f"구간 {seg} 혈당 및 식사량",
                xaxis=dict(
                    title="시간 (시:분)",
                    tickformat="%H:%M",
                    tickangle=45
                ),
                yaxis=dict(
                    title="혈당 (mg/dL)",
                    range=[40, max(result) + 20],
                    side="left"
                ),
                yaxis2=dict(
                    title="식사량 (g)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    range=[0, max(meal_series) + 10 if any(meal_series) else 10]
                ),
                legend=dict(x=0.01, y=1.1, orientation="h"),
                bargap=0.1
            )

            # 📊 렌더링
            st.plotly_chart(fig, use_container_width=True)

            # 📊 혈당 결과 해석
            st.markdown("#### 🔍 혈당 결과 해석")
            bg_final = result[-1]
            if bg_final < 70:
                st.warning(f"⚠️ 최종 혈당이 {bg_final:.1f} mg/dL로 저혈당입니다. 인슐린 용량을 줄여보세요.")
            elif bg_final > 180:
                st.warning(f"⚠️ 최종 혈당이 {bg_final:.1f} mg/dL로 고혈당입니다. 인슐린 용량을 늘려볼 수 있습니다.")
            else:
                st.success(f"✅ 최종 혈당 {bg_final:.1f} mg/dL — 안정적인 범위입니다.")

            events, _ = analyze_glucose_events(result, time_range)
            st.subheader("🩸 혈당 이상 구간 요약")
            if events:
                for msg in events:
                    st.markdown(msg)
            else:
                st.success("✅ 모든 시간대에서 혈당이 정상 범위(70~180 mg/dL)를 유지했습니다.")

            
        # if st.button(f"🔁 구간 {seg} 다시 설정"):
        #     if bg_key in st.session_state:
        #         del st.session_state[bg_key]

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

    # 샘플 타임스탬프 생성 (3분 간격, 총 480개: 24시간 분량)
    start_time = datetime.datetime.strptime("00:00", "%H:%M")
    time_range = [start_time + datetime.timedelta(minutes=3 * i) for i in range(480)]

    # 데이터 로드
    df = pd.read_csv(f"data/{st.session_state.csv_file}")
    df["Time"] = pd.to_datetime(df["Time"])

    for i in range(1, 4):
        bg_key = f"bg_user{i}"
        dose_key = f"dose{i}"
        if bg_key in st.session_state:
            full_bg.extend(st.session_state[bg_key])
            dose_value = st.session_state.get(dose_key, 0.0)
            full_bolus.extend([dose_value] * 160)
            full_basal.extend([st.session_state.get("dose_basal", 0.02)] * 160)

    meal_total = df.iloc[:480]["CHO"].sum()

    # 요약 생성
    st.markdown("### 📊 오늘의 혈당 제어 요약")
    st.success(summarize_today(full_basal, full_bolus, meal_total, full_bg))

    # AI 제어 결과 불러오기
    ai_df = st.session_df.iloc[:480].reset_index(drop=True)
    ai_bg = ai_df["BG"].tolist()

    # 혈당 비교 시각화
    fig = go.Figure()
        
    # ✅ 1. 정상 범위 음영 (70~180 mg/dL)
    fig.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=time_range[0], x1=time_range[-1],
        y0=70, y1=180,
        fillcolor="green",
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    # 사용자 혈당
    fig.add_trace(go.Scatter(
        x=time_range,
        y=full_bg,
        mode="lines",
        name="사용자 제어 혈당",
        line=dict(color="blue")
    ))

    # AI 혈당
    fig.add_trace(go.Scatter(
        x=time_range,
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

    # 레이아웃 설정
    fig.update_layout(
        title="AI vs 사용자 제어 시뮬레이션 결과 (24시간, 3분 간격)",
        xaxis_title="시간 (HH:MM)",
        yaxis_title="혈당 (mg/dL)",
        xaxis=dict(
            tickformat="%H:%M",
            tickangle=45
        ),
        legend=dict(x=0, y=1.1, orientation="h")
        )
    
    st.plotly_chart(fig, use_container_width=True)    

    # 3. TIR 계산 및 막대 시각화
    def compute_tir(bg_series):
        in_range = np.logical_and(np.array(bg_series) >= 70, np.array(bg_series) <= 180)
        return 100 * np.sum(in_range) / len(bg_series)

    tir_ai = compute_tir(ai_bg)
    tir_user = compute_tir(full_bg)

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
    st.subheader("📊 TIR (Time in Range: 70~180 mg/dL)")
    st.write(f"✅ **AI TIR**: {tir_ai:.2f}%")
    st.write(f"🧑‍⚕️ **사용자 TIR**: {tir_user:.2f}%")

    # 결과 비교 메시지
    st.subheader("🏁 결과 요약")
    if tir_user > tir_ai:
        st.success("🎉 **축하합니다!** 사용자 제어가 AI보다 높은 TIR을 기록했습니다!")
    elif tir_user < tir_ai:
        st.error("🤖 아쉽습니다. AI 제어가 더 높은 TIR을 기록했습니다.")
    else:
        st.info("⚖️ 사용자와 AI가 동일한 TIR 성능을 보여주었습니다.")

    def compute_variability(bg_series):
        bg_array = np.array(bg_series)
        avg = np.mean(bg_array)
        std = np.std(bg_array)
        cv = (std / avg) * 100
        return avg, std, cv
    
    # 계산
    avg_ai, std_ai, cv_ai = compute_variability(ai_bg)
    avg_user, std_user, cv_user = compute_variability(full_bg)

    # 표 형태 요약
    st.subheader("📊 혈당 변동성 비교")

    st.markdown(f"""
    | 구분 | 평균 혈당 | 표준편차 (SD) | 변동계수 (CV%) |
    |------|------------|----------------|----------------|
    | **AI** | {avg_ai:.1f} mg/dL | {std_ai:.1f} | {cv_ai:.1f}% |
    | **사용자** | {avg_user:.1f} mg/dL | {std_user:.1f} | {cv_user:.1f}% |
    """, unsafe_allow_html=True)

    # 해석 메시지
    st.markdown("#### 🔍 혈당 변동 해석")
    if cv_user > cv_ai:
        st.warning(f"⚠️ 사용자의 혈당 변동성이 더 큽니다. (CV {cv_user:.1f}% > {cv_ai:.1f}%)")
    else:
        st.success(f"✅ 사용자의 혈당 변동성이 더 낮아 안정적인 패턴을 보였습니다. (CV {cv_user:.1f}% < {cv_ai:.1f}%)")

    # 고위험 경고
    if cv_user > 36:
        st.error("🚨 혈당 변동계수(CV)가 36%를 초과해 고위험군에 해당할 수 있습니다.")


    if st.button("✅ 시뮬레이션 완료 → 처음으로"):
        for key in list(st.session_state.keys()):
            if key.startswith("bg_user") or key.startswith("env_") or key.startswith("dose"):
                del st.session_state[key]
        st.session_state.step = 0
        st.rerun()