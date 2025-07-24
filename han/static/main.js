// main.js
// [외국인 방문객 예측 서비스] 메인 JS (그래프, 뉴스, UI 제어 전담)
// 2025-07-24: 체크박스 연동 방식 완전 변경 (체크시만 그래프 표시, y축 auto)

let allCountries = [], allPurposes = [],
    colors = ['#007bff','#e94e77','#44b39d','#ffc857','#3a7ca5','#7fcd91','#ae5a41','#d72660','#479761','#e4b363'],
    lastPredictResults = null, lastCombos = null, currentCheckboxState = {};

// 드롭다운 옵션 채우기
const setSelectOptions = (id, list, def) => {
    const el = document.getElementById(id);
    el.innerHTML = list.map(v => `<option value="${v}">${v}</option>`).join('');
    if (def !== undefined) el.value = def;
};

// 국가/목적 목록 서버에서 로드
const fetchCountries = () =>
    fetch('/api/countries').then(res => res.json()).then(res => {
        // "러시아(연방)" 혹시라도 있으면 제거
        allCountries = res.filter(c => c !== "러시아(연방)");
        setSelectOptions('country1', ["전체", ...allCountries]);
    });
const fetchPurposes = () =>
    fetch('/api/purposes').then(res => res.json()).then(res => {
        allPurposes = res.filter(x => x !== "기타");
        let order = ['전체', '관광', '유학연수', '상용', '공용'];
        setSelectOptions('purpose1', order.filter(x => x === "전체" || allPurposes.includes(x)));
    });

// 연/월 선택기: 2025-06~2026-12만 허용, 연도 선택시 월 자동제어
const setYearMonthSelect = () => {
    const years = [2025, 2026];
    setSelectOptions('start-year', years, 2025);
    setSelectOptions('end-year', years, 2025);

    // 동적으로 월 옵션 변경
    const setMonthOpts = (year, selectId, defMonth) => {
        let minMonth = (parseInt(year) === 2025) ? 6 : 1;
        let maxMonth = 12;
        let arr = [];
        for(let i = minMonth; i <= maxMonth; ++i) arr.push(i);
        setSelectOptions(selectId, arr, defMonth || minMonth);
    };

    setMonthOpts(2025, 'start-month', 6);
    setMonthOpts(2025, 'end-month', 12);

    document.getElementById('start-year').onchange = function() {
        setMonthOpts(this.value, 'start-month');
    };
    document.getElementById('end-year').onchange = function() {
        setMonthOpts(this.value, 'end-month');
    };
};

window.onload = () => {
    fetchCountries();
    fetchPurposes();
    setYearMonthSelect();

    document.getElementById('compare-btn').onclick = onCompareBtnClick;
    document.getElementById('show-detail-btn').onclick = toggleDetailSummary;

    document.getElementById('menu-news').onclick = () => {
        document.getElementById('news-section').style.display = "";
        document.getElementById('predict-section').style.display = "none";
        loadNewsList();
    };
    document.getElementById('menu-predict').onclick = () => {
        document.getElementById('news-section').style.display = "none";
        document.getElementById('predict-section').style.display = "";
    };
};

// 예측 버튼 클릭: 조건 조합 구성, API 요청, 결과 그래프 표시
function onCompareBtnClick() {
    const c = document.getElementById('country1').value, p = document.getElementById('purpose1').value,
    combos =
        c === "전체" && p === "전체" ? [...allPurposes.map(pu => ({country: "전체", purpose: pu})), {country: "전체", purpose: "전체"}] :
        c === "전체" ? [{country: "전체", purpose: p}, {country: "전체", purpose: "전체"}] :
        p === "전체" ? [...allPurposes.map(pu => ({country: c, purpose: pu})), {country: c, purpose: "전체"}] :
        [{country: c, purpose: p}, {country: c, purpose: "전체"}];

    const startYM = `${document.getElementById('start-year').value}-${String(document.getElementById('start-month').value).padStart(2, '0')}`;
    const endYM = `${document.getElementById('end-year').value}-${String(document.getElementById('end-month').value).padStart(2, '0')}`;

    fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({combos, start_ym: startYM, end_ym: endYM})
    })
    .then(res => res.json())
    .then(res => {
        lastPredictResults = res.results;
        lastCombos = combos;
        let p = document.getElementById('purpose1').value;
        const defaultPurposes = p === "전체" ? ["전체", ...allPurposes] : ["전체", p];
        defaultPurposes.forEach(x => currentCheckboxState[x] = true);
        drawGraphWithCheckbox(res.results, 'compare-graph', 'compare-title', combos, defaultPurposes);
        document.getElementById('detail-box').style.display = "none";
        document.getElementById('show-detail-btn').textContent = "자세히";
    });
}

// x축 1/3 압축 구현
function drawGraphWithCheckbox(results, divId, titleId, combos, checkedPurposes) {
    let traces = [], colorIdx = 0, xticks = [], all_x = [];
    const c = document.getElementById('country1').value,
          p = document.getElementById('purpose1').value,
          checkboxPurposes = p === "전체" ? ["전체", ...allPurposes] : ["전체", p];
    const showPurposes = checkedPurposes && checkedPurposes.length > 0 ? checkedPurposes : checkboxPurposes;

    // 사용자가 선택한 예측 시작/종료 시점
    const userStartYM = `${document.getElementById('start-year').value}-${String(document.getElementById('start-month').value).padStart(2, '0')}`;
    const userEndYM = `${document.getElementById('end-year').value}-${String(document.getElementById('end-month').value).padStart(2, '0')}`;

    // 압축 대상 3개월 단위 구간(고정)
    const q3_months = [
        "2021-06","2021-09","2021-12","2022-03","2022-06","2022-09","2022-12","2023-03","2023-06","2023-09","2023-12","2024-03","2024-06"
    ];
    let compressed_xmap = {}, compressed_xticks = [], displayXTicks = [];
    let compressedLen = q3_months.length;
    let expandLen = 0;

    showPurposes.forEach(purpose => {
        let r = (c === "전체")
            ? results.find(res => res.country === "전체" && res.purpose === purpose)
            : results.find(res => res.country === c && res.purpose === purpose);
        if (!r) return;
        
        let yArr = (r.values || []).map(v => isNaN(v) || v == null ? 0 : v);
        if (yArr.length === 0) return;

        let orig_x = r.yms;
        let isActualArr = r.is_actual;
        
        xticks = orig_x;
        all_x = orig_x;

        // x축 압축 좌표계 설정 (기존과 동일)
        let expand_idx = [];
        let xvals = [];
        orig_x.forEach((d, idx) => {
            if(q3_months.includes(d)) {
                let rate = idx / (compressedLen-1);
                xvals.push(rate * (1/3));
                compressed_xmap[d] = rate * (1/3);
            } else {
                expand_idx.push(idx);
            }
        });
        expandLen = orig_x.length - compressedLen;
        orig_x.forEach((d, idx) => {
            if(!q3_months.includes(d)) {
                let rate = (idx-compressedLen) / (expandLen-1>0 ? expandLen-1:1);
                let val = (1/3) + rate * (2/3);
                xvals[idx] = val;
                compressed_xmap[d] = val;
            }
        });

        // ============ 구간 분리 로직 ============
        // 1. 실제 데이터 구간 찾기
        let realStartIdx = 0;
        let realEndIdx = -1;
        for (let i = 0; i < isActualArr.length; i++) {
            if (isActualArr[i] === true) {
                realEndIdx = i;
            }
        }

        // 2. 사용자 검색 구간 찾기
        let userSearchStartIdx = -1;
        let userSearchEndIdx = -1;
        let userPredictYms = r.predict_yms || [];
        
        for (let i = 0; i < orig_x.length; i++) {
            if (userPredictYms.includes(orig_x[i])) {
                if (userSearchStartIdx === -1) userSearchStartIdx = i;
                userSearchEndIdx = i;
            }
        }

        // 3. 중간 예측 구간 (2025-06 ~ 사용자 검색 시작 이전)
        let middleStartIdx = realEndIdx + 1;
        let middleEndIdx = (userSearchStartIdx > 0) ? userSearchStartIdx - 1 : -1;

        // 중간 구간이 존재하는지 체크
        let hasMiddleSection = (middleStartIdx >= 0 && middleEndIdx >= middleStartIdx);

        // ===== 그래프 그리기 =====
        
        // 1) 실제 데이터 구간 (실선)
        if (realEndIdx >= realStartIdx) {
            traces.push({
                x: xvals.slice(realStartIdx, realEndIdx + 1),
                y: yArr.slice(realStartIdx, realEndIdx + 1),
                name: purpose,
                mode: 'lines+markers',
                line: {color: colors[colorIdx % colors.length], width: 3, dash: 'solid'},
                marker: {color: colors[colorIdx % colors.length], size: 9},
                opacity: 1,
                hovertemplate: `<b>${purpose}</b><br>%{customdata}<br>입국자: %{y:,}명<extra></extra>`,
                customdata: orig_x.slice(realStartIdx, realEndIdx + 1),
                showlegend: true
            });
        }

        // 2) 중간 예측 구간 (회색 점선) - 존재할 때만
        if (hasMiddleSection) {
            traces.push({
                x: xvals.slice(middleStartIdx, middleEndIdx + 1),
                y: yArr.slice(middleStartIdx, middleEndIdx + 1),
                name: purpose + " (중간 예측)",
                mode: 'lines+markers',
                line: {color: '#666666', width: 2, dash: 'dot'},
                marker: {color: '#666666', size: 7, symbol: 'circle-open'},
                opacity: 0.8,
                hovertemplate: `<b>${purpose} (중간 예측)</b><br>%{customdata}<br>입국자: %{y:,}명<extra></extra>`,
                customdata: orig_x.slice(middleStartIdx, middleEndIdx + 1),
                showlegend: false
            });
        }

        // 3) 사용자 검색 구간 (해당 색상의 점선)
        if (userSearchStartIdx >= 0 && userSearchEndIdx >= userSearchStartIdx) {
            traces.push({
                x: xvals.slice(userSearchStartIdx, userSearchEndIdx + 1),
                y: yArr.slice(userSearchStartIdx, userSearchEndIdx + 1),
                name: purpose + " (예측)",
                mode: 'lines+markers',
                line: {color: colors[colorIdx % colors.length], width: 3, dash: 'dot'},
                marker: {color: colors[colorIdx % colors.length], size: 9, symbol: 'circle-open'},
                opacity: 1,
                hovertemplate: `<b>${purpose} (예측)</b><br>%{customdata}<br>입국자: %{y:,}명<extra></extra>`,
                customdata: orig_x.slice(userSearchStartIdx, userSearchEndIdx + 1),
                showlegend: true
            });
        }

        // ============ 연결선 추가 ============
        
        if (hasMiddleSection) {
            // 경우 1: 실제값 + 예측값(회색) + 예측값(점선)
            
            // 1-1) 실제값 마지막 ↔ 중간 예측 첫째 (회색 점선 연결)
            if (realEndIdx >= 0 && middleStartIdx >= 0) {
                traces.push({
                    x: [xvals[realEndIdx], xvals[middleStartIdx]],
                    y: [yArr[realEndIdx], yArr[middleStartIdx]],
                    name: purpose + " (실제-중간 연결)",
                    mode: 'lines',
                    line: {color: '#888888', width: 1.5, dash: 'dot'},
                    opacity: 0.6,
                    hoverinfo: 'skip',
                    showlegend: false
                });
            }
            
            // 1-2) 중간 예측 마지막 ↔ 사용자 검색 첫째 (회색 점선 연결)
            if (middleEndIdx >= 0 && userSearchStartIdx >= 0) {
                traces.push({
                    x: [xvals[middleEndIdx], xvals[userSearchStartIdx]],
                    y: [yArr[middleEndIdx], yArr[userSearchStartIdx]],
                    name: purpose + " (중간-사용자 연결)",
                    mode: 'lines',
                    line: {color: '#888888', width: 1.5, dash: 'dot'},
                    opacity: 0.6,
                    hoverinfo: 'skip',
                    showlegend: false
                });
            }
            
        } else {
            // 경우 2: 실제값 + 예측값(점선) - 중간 구간이 없는 경우
            
            // 2-1) 실제값 마지막 ↔ 사용자 검색 첫째 (해당 색상의 점선 연결)
            if (realEndIdx >= 0 && userSearchStartIdx >= 0) {
                traces.push({
                    x: [xvals[realEndIdx], xvals[userSearchStartIdx]],
                    y: [yArr[realEndIdx], yArr[userSearchStartIdx]],
                    name: purpose + " (실제-예측 연결)",
                    mode: 'lines',
                    line: {color: colors[colorIdx % colors.length], width: 2, dash: 'dot'},
                    opacity: 0.7,
                    hoverinfo: 'skip',
                    showlegend: false
                });
            }
        }

        colorIdx++;
    });

    // 나머지 그래프 설정은 기존과 동일...
    compressed_xticks = all_x.map(d => compressed_xmap[d]);
    
    // 코로나 구간 설정
    let covidEnd = "2022-05";
    let covidX1 = null;
    for (let i = 0; i < all_x.length; i++) {
        if (all_x[i] > covidEnd) {
            covidX1 = compressed_xmap[all_x[i-1]];
            break;
        }
    }
    if (covidX1 === null) covidX1 = compressed_xmap[all_x[all_x.length-1]];

    displayXTicks = all_x.map(d => {
        if (d >= "2021-06" && d <= "2022-05") return `<span style="color:#d72660; font-weight:bold">${d.slice(0,4)}년 ${d.slice(5)}월</span>`;
        return `${d.slice(0,4)}년 ${d.slice(5)}월`;
    });

    let yMax = 1;
    traces.forEach(t => { if (t.y && t.y.length) yMax = Math.max(yMax, ...t.y); });

    let covidRegions = [{
        type: 'rect', xref: 'x', yref: 'paper',
        x0: compressed_xmap["2021-06"], x1: covidX1, y0: 0, y1: 1,
        fillcolor: '#ffe0e0', opacity: 0.35, line: {width: 0}, layer: 'below'
    }];

    Plotly.newPlot(divId, traces, {
        margin: {t: 80, r: 60, l: 80, b: 100},
        xaxis: {
            tickangle: 45, showgrid: true, title: {text: "날짜", standoff: 35},
            tickmode: 'array', tickvals: compressed_xticks, ticktext: displayXTicks, automargin: true,
            range: [0, 1]
        },
        yaxis: {title: '입국자수', rangemode: "tozero", range: [0, Math.ceil(yMax * 1.03)], tickformat: ",d", height: 600, ticksuffix: "명"},
        hovermode: 'closest', shapes: covidRegions,
        legend: {orientation: "h", x: 0.5, xanchor: "center", y: 1.20, font: {size: 14}},
        plot_bgcolor: "#fff", paper_bgcolor: "#fff",
        xaxis_showline: true, xaxis_linecolor: "#3a7ca5", xaxis_linewidth: 2,
        yaxis_showline: true, yaxis_linecolor: "#3a7ca5", yaxis_linewidth: 2,
    }, {responsive: true, displayModeBar: true, displaylogo: false});
    
    drawPurposeCheckboxes(checkboxPurposes, showPurposes);
}

// 체크박스 UI 렌더링 및 이벤트 (checked만 그래프에)
function drawPurposeCheckboxes(purposeList, checkedPurposes = null) {
    const area = document.getElementById('purpose-checkboxes');
    area.innerHTML = '';
    purposeList.forEach((p, idx) => {
        const id = `purpose-cb-${idx}`;
        const checked = checkedPurposes ? checkedPurposes.includes(p) : true;
        area.innerHTML += `<label><input type="checkbox" ${checked ? "checked" : ""} id="${id}" data-purpose="${p}">${p === '전체' ? '전체' : `<span style="font-size:0.9em">${p}</span>`}</label>`;
        currentCheckboxState[p] = checked;
    });
    purposeList.forEach((p, idx) => {
        document.getElementById(`purpose-cb-${idx}`).onchange = function() {
            currentCheckboxState[p] = this.checked;
            // 현재 체크된 목적만 수집
            const checked = purposeList.filter(pp => currentCheckboxState[pp]);
            drawGraphWithCheckbox(lastPredictResults, 'compare-graph', 'compare-title', lastCombos, checked);
        };
    });
}

// 자세히 버튼 토글
const toggleDetailSummary = () => {
    let detailDiv = document.getElementById('detail-box'),
        btn = document.getElementById('show-detail-btn');
    if (detailDiv.style.display === "block") {
        detailDiv.style.display = "none";
        btn.textContent = "자세히";
    } else {
        showDetailSummary();
        btn.textContent = "닫기";
    }
};

// 예측 결과 통계/신뢰도 요약 박스 표시 (표 방식)
const showDetailSummary = () => {
    let detailDiv = document.getElementById('detail-box');
    detailDiv.innerHTML = '';
    if (!lastPredictResults || lastPredictResults.length === 0) {
        detailDiv.style.display = "block";
        detailDiv.innerHTML = "<div class='alert alert-warning'>먼저 비교 버튼으로 그래프를 그려주세요.</div>";
        return;
    }
    const c = document.getElementById('country1').value,
          p = document.getElementById('purpose1').value,
          isCountryAll = (c === "전체"), isPurposeAll = (p === "전체");
    let r2list = [], mapelist = [], conflist = [];
    lastPredictResults.forEach(r => {
        if (isCountryAll && isPurposeAll) {
            if (r.r2 != null) r2list.push(r.r2);
            if (r.mape != null) mapelist.push(r.mape);
            if (r.confidence != null) conflist.push(r.confidence);
        } else if (!isCountryAll && isPurposeAll) {
            if (r.country === c) {
                if (r.r2 != null) r2list.push(r.r2);
                if (r.mape != null) mapelist.push(r.mape);
                if (r.confidence != null) conflist.push(r.confidence);
            }
        } else if (isCountryAll && !isPurposeAll) {
            if (r.purpose === p) {
                if (r.r2 != null) r2list.push(r.r2);
                if (r.mape != null) mapelist.push(r.mape);
                if (r.confidence != null) conflist.push(r.confidence);
            }
        } else {
            if (r.country === c && r.purpose === p) {
                if (r.r2 != null) r2list.push(r.r2);
                if (r.mape != null) mapelist.push(r.mape);
                if (r.confidence != null) conflist.push(r.confidence);
            }
        }
    });
    // 통계 함수
    const arrStat = arr => {
        arr = arr.flatMap(v => Array.isArray(v) ? v : [v]).filter(x => x != null && !isNaN(x));
        const avg = arr.length ? arr.reduce((s,x)=>s+x,0)/arr.length : null;
        return {
            min: arr.length ? Math.min(...arr) : null,
            mean: avg,
            max: arr.length ? Math.max(...arr) : null
        };
    };
    const r2stat = arrStat(r2list), mapestat = arrStat(mapelist), confstat = arrStat(conflist);
    const evalR2 = r2 => r2 == null ? "-" : r2 >= 0.9 ? "매우 좋음" : r2 >= 0.7 ? "보통" : "주의(신뢰 낮음)";
    const evalMape = m => m == null ? "-" : m <= 10 ? "매우 좋음" : m <= 20 ? "보통" : "주의(오차 큼)";
    const evalConf = c => c == null ? "-" : c >= 90 ? "매우 좋음" : c >= 80 ? "보통" : "주의(신뢰 낮음)";
    detailDiv.style.display = "block";
    detailDiv.innerHTML =
        `<div class="card p-3">
            <b>예측 구간 성능 요약</b>
            <div class="table-responsive mb-2">
            <table class="table table-bordered table-sm text-center align-middle mb-0">
                <thead class="table-light">
                    <tr>
                        <th>지표</th>
                        <th>최소</th>
                        <th>평균</th>
                        <th>최대</th>
                        <th>평가</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>r2</b><br>(설명력)</td>
                        <td>${r2stat.min === null ? "-" : r2stat.min.toFixed(4)}</td>
                        <td>${r2stat.mean === null ? "-" : r2stat.mean.toFixed(4)}</td>
                        <td>${r2stat.max === null ? "-" : r2stat.max.toFixed(4)}</td>
                        <td>${evalR2(r2stat.mean)}</td>
                    </tr>
                    <tr>
                        <td><b>MAPE</b><br>(평균예측오차)</td>
                        <td>${mapestat.min === null ? "-" : mapestat.min.toFixed(2)}%</td>
                        <td>${mapestat.mean === null ? "-" : mapestat.mean.toFixed(2)}%</td>
                        <td>${mapestat.max === null ? "-" : mapestat.max.toFixed(2)}%</td>
                        <td>${evalMape(mapestat.mean)}</td>
                    </tr>
                    <tr>
                        <td><b>Confidence</b><br>(신뢰도)</td>
                        <td>${confstat.min === null ? "-" : confstat.min.toFixed(1)}</td>
                        <td>${confstat.mean === null ? "-" : confstat.mean.toFixed(1)}</td>
                        <td>${confstat.max === null ? "-" : confstat.max.toFixed(1)}</td>
                        <td>${evalConf(confstat.mean)}</td>
                    </tr>
                </tbody>
            </table>
            </div>
            <span class="text-secondary small">* 평가는 평균값 기준 (모델/기간마다 달라질 수 있음)</span>
        </div>`;
};
