// main.js
// [외국인 방문객 예측 서비스] 메인 JS (그래프, 뉴스, UI 제어 전담)
// 주석은 보존하며, 로직은 최대한 간결하게 단축

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
        allCountries = res;
        setSelectOptions('country1', ["전체", ...allCountries]);
    });
const fetchPurposes = () =>
    fetch('/api/purposes').then(res => res.json()).then(res => {
        allPurposes = res.filter(x => x !== "기타"); // 기타 제외
        setSelectOptions('purpose1', ["전체", ...allPurposes]);
    });

// 연/월 선택기 초기화
const setYearMonthSelect = (startYearId, startMonthId, endYearId, endMonthId, minYear, maxYear) => {
    const years = Array.from({length: maxYear - minYear + 1}, (_, i) => minYear + i);
    const months = Array.from({length: 12}, (_, i) => i + 1);
    setSelectOptions(startYearId, years, 2025);
    setSelectOptions(endYearId, years, 2025);
    setSelectOptions(startMonthId, months, 1);
    setSelectOptions(endMonthId, months, 12);
};

// 메인 진입점
window.onload = () => {
    fetchCountries();
    fetchPurposes();
    setYearMonthSelect('start-year','start-month','end-year','end-month',2005,2026);

    document.getElementById('compare-btn').onclick = onCompareBtnClick;
    document.getElementById('show-detail-btn').onclick = toggleDetailSummary;
    document.getElementById('covid-visual').onchange = function() {
        for (let t of document.getElementsByClassName('covid-region'))
            t.style.display = (this.value === "show") ? "" : "none";
    };
    // 뉴스/예측 탭 전환
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

// 뉴스 리스트 로딩 (페이지네이션 포함)
const loadNewsList = (page = 1) => {
    fetch('/api/news?page=' + page)
        .then(res => res.json())
        .then(data => {
            let list = document.getElementById('news-list');
            list.innerHTML = '';
            if (!data.news || data.news.length === 0) {
                list.innerHTML = "<li class='list-group-item'>뉴스가 없습니다.</li>";
                document.getElementById('news-pagination').innerHTML = '';
                return;
            }
            data.news.forEach(item => {
                list.innerHTML += `<li class='list-group-item'>
                    <a href="${item.link}" target="_blank">${item.title}</a>
                    <br><small>${item.pubDate}</small>
                    <div class="text-muted small">${item.description}</div>
                </li>`;
            });
            // 페이지네이션 UI
            let total = data.news_total || data.news.length, pageCount = Math.ceil(total / 20), curr = page, nav = '';
            if (curr > 1)
                nav += `<li class="page-item"><a class="page-link" href="#" onclick="loadNewsList(${curr-1});return false;">이전</a></li>`;
            for (let i = 1; i <= pageCount; i++)
                nav += `<li class="page-item ${i == curr ? 'active' : ''}"><a class="page-link" href="#" onclick="loadNewsList(${i});return false;">${i}</a></li>`;
            if (curr < pageCount)
                nav += `<li class="page-item"><a class="page-link" href="#" onclick="loadNewsList(${curr+1});return false;">다음</a></li>`;
            document.getElementById('news-pagination').innerHTML = nav;
        });
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
        drawGraphWithCheckbox(res.results, 'compare-graph', 'compare-title', combos);
        document.getElementById('detail-box').style.display = "none";
        document.getElementById('show-detail-btn').textContent = "자세히";
    });
}

// 예측 결과 그래프 + 체크박스 동기화
function drawGraphWithCheckbox(results, divId, titleId, combos) {
    let traces = [], colorIdx = 0, xticks = [], yMin = null, yMax = null;
    const c = document.getElementById('country1').value,
          p = document.getElementById('purpose1').value,
          checkboxPurposes = p === "전체" ? ["전체", ...allPurposes] : ["전체", p];

    checkboxPurposes.forEach(purpose => {
        let r = (c === "전체")
            ? results.find(res => res.country === "전체" && res.purpose === purpose)
            : results.find(res => res.country === c && res.purpose === purpose);
        if (!r) return;
        let yArr = (r.values || []).map(v => isNaN(v) || v == null ? 0 : v);
        if (yArr.length === 0) return;
        let isActual = r.is_actual || [], splitIdx = isActual.findIndex(a => a === false),
            color = colors[colorIdx++ % colors.length], traceName = purpose;
        if (splitIdx !== -1 && splitIdx > 0) {
            let xBefore = r.yms.slice(0, splitIdx + 1), yBefore = yArr.slice(0, splitIdx + 1),
                xAfter = r.yms.slice(splitIdx), yAfter = yArr.slice(splitIdx);
            traces.push({
                x: xBefore, y: yBefore, name: traceName, mode: 'lines+markers',
                line: {color, width: 3, dash: 'solid'}, marker: {color, size: 9}, opacity: 1,
                customdata: Array(xBefore.length).fill(traceName),
                hovertemplate: `<b>${traceName}</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`, showlegend: true
            });
            traces.push({
                x: [r.yms[splitIdx - 1], r.yms[splitIdx]],
                y: [yArr[splitIdx - 1], yArr[splitIdx]],
                name: traceName + " (연결)", mode: 'lines+markers',
                line: {color, width: 3, dash: 'dot'}, marker: {color, size: 9}, opacity: 1,
                customdata: Array(2).fill(traceName + " (연결)"),
                hovertemplate: `<b>${traceName} (연결)</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`, showlegend: false
            });
            traces.push({
                x: xAfter, y: yAfter, name: traceName + " (예측)", mode: 'lines+markers',
                line: {color, width: 3, dash: 'dot'}, marker: {color, size: 9, symbol: 'circle-open'}, opacity: 1,
                customdata: Array(xAfter.length).fill(traceName + " (예측)"),
                hovertemplate: `<b>${traceName} (예측)</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`, showlegend: true
            });
        } else {
            traces.push({
                x: r.yms, y: yArr, name: traceName, mode: 'lines+markers',
                line: {color, width: 3, dash: 'solid'}, marker: {color, size: 9}, opacity: 1,
                customdata: Array(r.yms.length).fill(traceName),
                hovertemplate: `<b>${traceName}</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`, showlegend: true
            });
        }
        xticks = r.yms;
        let ymin = Math.min(...yArr), ymax = Math.max(...yArr);
        if (yMin === null || ymin < yMin) yMin = ymin;
        if (yMax === null || ymax > yMax) yMax = ymax;
    });

    // x축, y축, 코로나기간 표시 등 그래프 옵션
    let displayXTicks = xticks.map(d => { let [y, m] = d.split('-'); return `${y}년 ${m}월`; });
    let covidRegions = [];
    if (document.getElementById('covid-visual').value === 'show')
        covidRegions.push({
            type: 'rect', xref: 'x', yref: 'paper',
            x0: '2020-03', x1: '2022-10', y0: 0, y1: 1,
            fillcolor: '#ffe0e0', opacity: 0.35, line: {width: 0}, layer: 'below'
        });
    Plotly.newPlot(divId, traces, {
        margin: {t: 80, r: 60, l: 80, b: 100},
        xaxis: {tickangle: 45, showgrid: true, title: {text: "날짜", standoff: 35}, tickmode: 'array', tickvals: xticks, ticktext: displayXTicks, automargin: true},
        yaxis: {title: '입국자수', rangemode: "tozero", range: [0, Math.ceil(yMax * 1.03)], tickformat: ",d", height: 600, ticksuffix: "명"},
        hovermode: 'closest', shapes: covidRegions,
        legend: {orientation: "h", x: 0.5, xanchor: "center", y: 1.20, font: {size: 14}},
        plot_bgcolor: "#fff", paper_bgcolor: "#fff",
        xaxis_showline: true, xaxis_linecolor: "#3a7ca5", xaxis_linewidth: 2,
        yaxis_showline: true, yaxis_linecolor: "#3a7ca5", yaxis_linewidth: 2,
    }, {responsive: true, displayModeBar: true, displaylogo: false});
    drawPurposeCheckboxes(checkboxPurposes);
}

// 체크박스 UI 렌더링 및 그래프 opacity 조정
function drawPurposeCheckboxes(purposeList) {
    const area = document.getElementById('purpose-checkboxes');
    area.innerHTML = '';
    purposeList.forEach((p, idx) => {
        const id = `purpose-cb-${idx}`;
        area.innerHTML += `<label><input type="checkbox" checked id="${id}" data-purpose="${p}">${p === '전체' ? '전체' : `<span style="font-size:0.9em">${p}</span>`}</label>`;
        currentCheckboxState[p] = true;
    });
    purposeList.forEach((p, idx) => {
        document.getElementById(`purpose-cb-${idx}`).onchange = function() {
            currentCheckboxState[p] = this.checked;
            const graphDiv = document.getElementById('compare-graph');
            const update = {opacity: graphDiv.data.map(t =>
                (t.customdata && t.customdata[0] && t.customdata[0].includes(p)) ? (this.checked ? 1 : 0.15) : (t.opacity || 1)
            )};
            Plotly.restyle(graphDiv, update, Array.from({length: update.opacity.length}, (_, i) => i));
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

// 예측 결과 통계/신뢰도 요약 박스 표시
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
          isCountryAll = (c === "전체"), isPurposeAll = (p === "전체"),
          filterRes = (fn) => lastPredictResults.filter(fn).flatMap(r => [r.r2, r.mape, r.confidence]);
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
    // 통계 함수 단일화
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

    // 평가 텍스트
    const evalR2 = r2 => r2 == null ? "-" : r2 >= 0.9 ? "매우 좋음" : r2 >= 0.7 ? "보통" : "주의(신뢰 낮음)";
    const evalMape = m => m == null ? "-" : m <= 10 ? "매우 좋음" : m <= 20 ? "보통" : "주의(오차 큼)";
    const evalConf = c => c == null ? "-" : c >= 90 ? "매우 좋음" : c >= 80 ? "보통" : "주의(신뢰 낮음)";

    detailDiv.style.display = "block";
    detailDiv.innerHTML =
        `<div class="card p-3">
        <b>예측 구간 성능 요약</b>
        <ul class="mb-2">
            <li><b>r2</b> (설명력): <span>최소: ${r2stat.min === null ? "-" : r2stat.min.toFixed(4)} / 평균: ${r2stat.mean === null ? "-" : r2stat.mean.toFixed(4)} / 최대: ${r2stat.max === null ? "-" : r2stat.max.toFixed(4)}</span> <span>→ ${evalR2(r2stat.mean)}</span></li>
            <li><b>mape</b> (평균예측오차): <span>최소: ${mapestat.min === null ? "-" : mapestat.min.toFixed(2)}% / 평균: ${mapestat.mean === null ? "-" : mapestat.mean.toFixed(2)}% / 최대: ${mapestat.max === null ? "-" : mapestat.max.toFixed(2)}%</span> <span>→ ${evalMape(mapestat.mean)}</span></li>
            <li><b>confidence</b> (신뢰도): <span>최소: ${confstat.min === null ? "-" : confstat.min.toFixed(1)} / 평균: ${confstat.mean === null ? "-" : confstat.mean.toFixed(1)} / 최대: ${confstat.max === null ? "-" : confstat.max.toFixed(1)}</span> <span>→ ${evalConf(confstat.mean)}</span></li>
        </ul>
        <span class="text-secondary small">* 평가는 평균값 기준 (모델/기간마다 달라질 수 있음)</span>
        </div>`;
};
