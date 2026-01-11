# Investment Analysis System & Ops Platform

이 저장소는 **미국 주식, 한국 주식, 배당 포트폴리오**를 분석하는 통합 투자 시스템과 이를 운영하기 위한 **Airflow 기반 Ops 플랫폼**을 포함합니다.

## 🏗 시스템 구성

### 1. 분석 엔진 (Core Engines)
각 모듈은 `implementations/` 디렉토리에 위치하며 독립적으로 실행 가능합니다.

| 모듈 | 디렉토리 | 기능 |
|------|----------|------|
| **🇺🇸 US Stock** | `implementations/USStockAnalysis` | S&P 500 스마트머니 분석, 수급 추적, AI 요약 |
| **🇰🇷 KR Stock** | `implementations/StockAI` | 국내 주식 파동 분석, 수급/뉴스 기반 AI 리포트 |
| **💰 Dividend** | `implementations/DividendOptimizer` | 월배당 포트폴리오 최적화, 백테스트 |

### 2. 운영 플랫폼 (Ops Platform)
Airflow와 Docker를 활용하여 분석 작업을 스케줄링하고 모니터링합니다.

- **Airflow**: 파이프라인 오케스트레이션 (매일 밤 자동 실행)
- **Docker Agent**: 격리된 환경에서 분석 코드 실행 (Docker-in-Docker)
- **PostgreSQL**: 메타데이터 관리

---

## 🚀 Ops 플랫폼 시작하기 (Airflow)

로컬 환경에서 Airflow를 띄우고 분석 파이프라인을 실행하는 방법입니다.

### 전제 조건
- Docker Desktop 설치 및 실행 중
- Python 3.10+ (옵션)

### 1. 환경 설정
`airflow` 디렉토리로 이동하여 환경 변수를 설정합니다.

```bash
cd airflow
cp .env.example .env
```
`.env` 파일을 열어 `AIRFLOW_HOST_PROJECT_PATH`를 현재 프로젝트의 절대 경로로 수정하세요.
(기본값은 `/Users/jshstorm/Documents/github/agent` 입니다)

### 2. 분석 에이전트 이미지 빌드
분석 코드를 실행할 Docker 이미지를 빌드합니다. (최초 1회)

```bash
# 프로젝트 루트에서 실행
docker build -t investment-analyst:latest -f docker/Dockerfile.analyst .
```

### 3. Airflow 실행
```bash
cd airflow
docker-compose up -d
```
약 1분 후 [http://localhost:8080](http://localhost:8080) 접속
- **ID**: `airflow`
- **PW**: `airflow`

### 4. 파이프라인 활성화
1. Airflow 웹 UI에서 `investment_daily_analysis` DAG를 찾습니다.
2. 왼쪽의 **ON/OFF 스위치**를 켜서 활성화합니다.
3. 오른쪽의 **▶ (Trigger DAG)** 버튼을 눌러 수동으로 즉시 실행해볼 수 있습니다.

---

## 💻 개발 가이드 (직접 실행)

Airflow 없이 개별 모듈을 직접 실행하거나 수정하려면 아래 가이드를 따르세요.

### 환경 설정
```bash
cd implementations
pip install -r requirements.txt
cp .env.example .env  # API 키 설정
```

### 실행 명령어
```bash
# 미국 주식 분석
cd implementations/USStockAnalysis
python update_all.py

# 한국 주식 분석
cd implementations/StockAI
python run_analysis.py

# 배당 최적화 웹서버
cd implementations/DividendOptimizer
python flask_app.py
```

---

## 📁 디렉토리 구조
```text
.
├── airflow/                  # Airflow 인프라 (Docker Compose, DAGs)
├── docker/                   # 분석 에이전트 Dockerfile
├── implementations/          # 핵심 분석 코드
│   ├── DividendOptimizer/    # 배당 최적화
│   ├── StockAI/              # 국내 주식
│   └── USStockAnalysis/      # 미국 주식
└── README.md                 # 메인 문서
```
