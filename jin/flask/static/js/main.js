let allCountries = [];
let allPurposes = [];
let colors = [
  '#007bff',
  '#e94e77',
  '#44b39d',
  '#ffc857',
  '#3a7ca5',
  '#7fcd91',
  '#ae5a41',
  '#d72660',
  '#479761',
  '#e4b363',
];
let lastPredictResults = null;
let lastCombos = null;
let currentCheckboxState = {};

function fetchCountries() {
  fetch('/api/countries')
    .then((res) => res.json())
    .then((res) => {
      allCountries = res;
      setSelectOptions('country1', ['전체', ...allCountries]);
    });
}
function fetchPurposes() {
  fetch('/api/purposes')
    .then((res) => res.json())
    .then((res) => {
      allPurposes = res;
      setSelectOptions('purpose1', ['전체', ...allPurposes]);
    });
}
function setSelectOptions(selectId, list) {
  let sel = document.getElementById(selectId);
  sel.innerHTML = '';
  list.forEach((v) => {
    let opt = document.createElement('option');
    opt.value = v;
    opt.text = v;
    sel.appendChild(opt);
  });
}
function setYearMonthSelect(startYearId, startMonthId, endYearId, endMonthId, minYear, minMonth, maxYear, maxMonth) {
  let years = [];
  for (let y = minYear; y <= maxYear; y++) years.push(y);
  setSelectOptions(startYearId, years);
  setSelectOptions(endYearId, years);

  let months = [];
  for (let m = 1; m <= 12; m++) months.push(m);
  setSelectOptions(startMonthId, months);
  setSelectOptions(endMonthId, months);

  document.getElementById(startYearId).value = 2025;
  document.getElementById(startMonthId).value = 1;
  document.getElementById(endYearId).value = 2025;
  document.getElementById(endMonthId).value = 12;
}

window.onload = function () {
  fetchCountries();
  fetchPurposes();
  setYearMonthSelect('start-year', 'start-month', 'end-year', 'end-month', 2005, 1, 2026, 12);

  document.getElementById('compare-btn').onclick = onCompareBtnClick;
  document.getElementById('show-detail-btn').onclick = toggleDetailSummary;
  document.getElementById('covid-visual').onchange = function () {
    let traces = document.getElementsByClassName('covid-region');
    let val = this.value;
    for (let t of traces) {
      t.style.display = val === 'show' ? '' : 'none';
    }
  };

  // === 뉴스 탭 클릭 이벤트 ===
  document.getElementById('menu-news').onclick = function () {
    document.getElementById('news-section').style.display = '';
    document.getElementById('predict-section').style.display = 'none';
    loadNewsList();
  };
  document.getElementById('menu-predict').onclick = function () {
    document.getElementById('news-section').style.display = 'none';
    document.getElementById('predict-section').style.display = '';
  };
};

// --- 뉴스 리스트 로드 함수 ---
function loadNewsList(page = 1) {
  fetch('/api/news?page=' + page)
    .then((res) => res.json())
    .then((data) => {
      let list = document.getElementById('news-list');
      list.innerHTML = '';
      if (!data.news || data.news.length === 0) {
        list.innerHTML = "<li class='list-group-item'>뉴스가 없습니다.</li>";
        document.getElementById('news-pagination').innerHTML = '';
        return;
      }
      // 뉴스 목록 출력
      data.news.forEach((item) => {
        list.innerHTML += `<li class='list-group-item'>
                    <a href="${item.link}" target="_blank">${item.title}</a>
                    <br><small>${item.pubDate}</small>
                    <div class="text-muted small">${item.description}</div>
                </li>`;
      });

      // --- 페이지네이션 ---  (여기가 바로 질문하신 코드!)
      let total = data.news_total || data.news.length;
      let pageCount = Math.ceil(total / 20);
      let curr = page;
      let nav = '';
      if (curr > 1)
        nav += `<li class="page-item"><a class="page-link" href="#" onclick="loadNewsList(${
          curr - 1
        });return false;">이전</a></li>`;
      for (let i = 1; i <= pageCount; i++) {
        nav += `<li class="page-item ${
          i == curr ? 'active' : ''
        }"><a class="page-link" href="#" onclick="loadNewsList(${i});return false;">${i}</a></li>`;
      }
      if (curr < pageCount)
        nav += `<li class="page-item"><a class="page-link" href="#" onclick="loadNewsList(${
          curr + 1
        });return false;">다음</a></li>`;
      document.getElementById('news-pagination').innerHTML = nav;
    });
}

function onCompareBtnClick() {
  let country = document.getElementById('country1').value;
  let purpose = document.getElementById('purpose1').value;
  let startYear = parseInt(document.getElementById('start-year').value);
  let startMonth = parseInt(document.getElementById('start-month').value);
  let endYear = parseInt(document.getElementById('end-year').value);
  let endMonth = parseInt(document.getElementById('end-month').value);

  let startYM = `${startYear}-${String(startMonth).padStart(2, '0')}`;
  let endYM = `${endYear}-${String(endMonth).padStart(2, '0')}`;

  let fetchCombos = [];
  let isCountryAll = country === '전체';
  let isPurposeAll = purpose === '전체';

  // 목적 전체가 아닌 경우에도 해당 국가의 전체 합 trace 포함(2개)
  if (!isCountryAll && !isPurposeAll) {
    fetchCombos.push({ country: country, purpose: purpose });
    fetchCombos.push({ country: country, purpose: '전체' });
  }
  // 국가 전체 + 목적 전체
  else if (isCountryAll && isPurposeAll) {
    for (const p of allPurposes) fetchCombos.push({ country: '전체', purpose: p });
    fetchCombos.push({ country: '전체', purpose: '전체' });
  }
  // 국가 전체 + 목적 단일
  else if (isCountryAll && !isPurposeAll) {
    fetchCombos.push({ country: '전체', purpose: purpose });
  }
  // 국가 단일 + 목적 전체
  else if (!isCountryAll && isPurposeAll) {
    for (const p of allPurposes) fetchCombos.push({ country: country, purpose: p });
    fetchCombos.push({ country: country, purpose: '전체' });
  }

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ combos: fetchCombos, start_ym: startYM, end_ym: endYM }),
  })
    .then((res) => res.json())
    .then((res) => {
      lastPredictResults = res.results;
      lastCombos = fetchCombos;
      drawGraphWithCheckbox(res.results, 'compare-graph', 'compare-title', fetchCombos);
      document.getElementById('detail-box').style.display = 'none';
      document.getElementById('show-detail-btn').textContent = '자세히';
    });
}

function drawGraphWithCheckbox(results, divId, titleId, combos) {
  let traces = [];
  let colorIdx = 0;
  let xticks = [];
  let yMin = null,
    yMax = null;
  let tracePurposeList = [];
  let traceKeys = new Set();
  let countrySelect = document.getElementById('country1');
  let purposeSelect = document.getElementById('purpose1');
  let mainCountry = countrySelect ? countrySelect.value : '';
  let mainPurpose = purposeSelect ? purposeSelect.value : '';
  let allPurposeNames = ['관광', '공용', '상용', '유학연수', '기타'];

  // 그래프 타이틀/목적명 폰트
  let titleHtml = '';
  if (mainPurpose === '전체') {
    titleHtml = `${mainCountry} / <b>전체</b> <span style="font-size:0.90em; color:#888; font-weight:400;">(${allPurposeNames.join(
      ', '
    )})</span>`;
  } else {
    titleHtml = `${mainCountry} / ${mainPurpose}`;
  }
  document.getElementById(titleId).innerHTML = titleHtml;

  // 목적별 체크박스 리스트(전체+목적)
  let graphPurposeList = ['전체', ...allPurposeNames];

  // trace를 목적 고정 순서로(전체, 공용, 관광, ...)
  let purposeOrder = ['전체', ...allPurposeNames];
  for (let purpose of purposeOrder) {
    let r = results.find(
      (res) =>
        res.purpose === purpose && (mainCountry === '전체' ? res.country === '전체' : res.country === mainCountry)
    );
    if (!r) continue;
    let yArr = (r.values || []).map((v) => (isNaN(v) || v == null ? 0 : v));
    if (yArr.length === 0) continue;
    let isActual = r.is_actual || [];
    let splitIdx = isActual.findIndex((a) => a === false);
    let color = colors[colorIdx % colors.length];
    colorIdx += 1;
    let traceName = purpose === '전체' ? '전체' : purpose;

    if (splitIdx !== -1 && splitIdx > 0) {
      let xBefore = r.yms.slice(0, splitIdx + 1);
      let yBefore = yArr.slice(0, splitIdx + 1);
      let xAfter = r.yms.slice(splitIdx);
      let yAfter = yArr.slice(splitIdx);

      traces.push({
        x: xBefore,
        y: yBefore,
        name: traceName,
        mode: 'lines+markers',
        line: { color: color, width: 3, dash: 'solid' },
        marker: { color: color, size: 9 },
        opacity: 1,
        customdata: Array(xBefore.length).fill(traceName),
        hovertemplate: `<b>${traceName}</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`,
        showlegend: true,
      });
      traces.push({
        x: [r.yms[splitIdx - 1], r.yms[splitIdx]],
        y: [yArr[splitIdx - 1], yArr[splitIdx]],
        name: traceName + ' (연결)',
        mode: 'lines+markers',
        line: { color: color, width: 3, dash: 'dot' },
        marker: { color: color, size: 9 },
        opacity: 1,
        customdata: Array(2).fill(traceName + ' (연결)'),
        hovertemplate: `<b>${traceName} (연결)</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`,
        showlegend: false,
      });
      traces.push({
        x: xAfter,
        y: yAfter,
        name: traceName + ' (예측)',
        mode: 'lines+markers',
        line: { color: color, width: 3, dash: 'dot' },
        marker: { color: color, size: 9, symbol: 'circle-open' },
        opacity: 1,
        customdata: Array(xAfter.length).fill(traceName + ' (예측)'),
        hovertemplate: `<b>${traceName} (예측)</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`,
        showlegend: true,
      });
    } else {
      traces.push({
        x: r.yms,
        y: yArr,
        name: traceName,
        mode: 'lines+markers',
        line: { color: color, width: 3, dash: 'solid' },
        marker: { color: color, size: 9 },
        opacity: 1,
        customdata: Array(r.yms.length).fill(traceName),
        hovertemplate: `<b>${traceName}</b><br>%{x}<br>입국자: %{y:,}명<extra></extra>`,
        showlegend: true,
      });
    }
    xticks = r.yms;
    let ymin = Math.min(...yArr),
      ymax = Math.max(...yArr);
    if (yMin === null || ymin < yMin) yMin = ymin;
    if (yMax === null || ymax > yMax) yMax = ymax;
  }

  let displayXTicks = xticks.map((d) => {
    let [y, m] = d.split('-');
    return `${y}년 ${m}월`;
  });

  let covidRegions = [];
  if (document.getElementById('covid-visual').value === 'show') {
    covidRegions.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: '2020-03',
      x1: '2022-10',
      y0: 0,
      y1: 1,
      fillcolor: '#ffe0e0',
      opacity: 0.35,
      line: { width: 0 },
      layer: 'below',
    });
  }

  let layout = {
    margin: { t: 80, r: 60, l: 80, b: 100 },
    xaxis: {
      tickangle: 45,
      showgrid: true,
      title: { text: '날짜', standoff: 35 },
      tickmode: 'array',
      tickvals: xticks,
      ticktext: displayXTicks,
      automargin: true,
    },
    yaxis: {
      title: '입국자수',
      rangemode: 'tozero',
      range: [0, Math.ceil(yMax * 1.03)],
      tickformat: ',d',
      height: 600,
      ticksuffix: '명',
    },
    hovermode: 'closest',
    shapes: covidRegions,
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.2, font: { size: 14 } },
    plot_bgcolor: '#fff',
    paper_bgcolor: '#fff',
    xaxis_showline: true,
    xaxis_linecolor: '#3a7ca5',
    xaxis_linewidth: 2,
    yaxis_showline: true,
    yaxis_linecolor: '#3a7ca5',
    yaxis_linewidth: 2,
  };

  Plotly.newPlot(divId, traces, layout, { responsive: true, displayModeBar: true, displaylogo: false });
  drawPurposeCheckboxes(graphPurposeList);
}

function drawPurposeCheckboxes(purposeList) {
  const area = document.getElementById('purpose-checkboxes');
  area.innerHTML = '';
  purposeList.forEach((p, idx) => {
    const id = `purpose-cb-${idx}`;
    let label = p === '전체' ? '전체' : `<span style="font-size:0.9em">${p}</span>`;
    area.innerHTML += `<label><input type="checkbox" checked id="${id}" data-purpose="${p}">${label}</label>`;
    currentCheckboxState[p] = true;
    setTimeout(() => {
      document.getElementById(id).onchange = function () {
        currentCheckboxState[p] = this.checked;
        let graphDiv = document.getElementById('compare-graph');
        let update = { opacity: [] };
        for (let i = 0; i < graphDiv.data.length; ++i) {
          let t = graphDiv.data[i];
          if (t.customdata && t.customdata[0] && t.customdata[0].indexOf(p) >= 0) {
            update.opacity.push(this.checked ? 1 : 0.15);
          } else {
            update.opacity.push(graphDiv.data[i].opacity || 1);
          }
        }
        Plotly.restyle(
          graphDiv,
          update,
          Array.from({ length: update.opacity.length }, (_, i) => i)
        );
      };
    }, 100);
  });
}

// 자세히 토글
function toggleDetailSummary() {
  let detailDiv = document.getElementById('detail-box');
  let btn = document.getElementById('show-detail-btn');
  if (detailDiv.style.display === 'block') {
    detailDiv.style.display = 'none';
    btn.textContent = '자세히';
  } else {
    showDetailSummary();
    btn.textContent = '닫기';
  }
}

function showDetailSummary() {
  let detailDiv = document.getElementById('detail-box');
  detailDiv.innerHTML = '';
  if (!lastCombos || lastCombos.length === 0) {
    detailDiv.style.display = 'block';
    detailDiv.innerHTML = "<div class='alert alert-warning'>먼저 비교 버튼으로 그래프를 그려주세요.</div>";
    return;
  }
  // 첫 번째 조합만 리포트 예시로 호출 (여러 조합 지원은 추후 확장)
  let combo = lastCombos[0];
  let startYear = document.getElementById('start-year').value;
  let startMonth = document.getElementById('start-month').value;
  let endYear = document.getElementById('end-year').value;
  let endMonth = document.getElementById('end-month').value;
  let startYM = `${startYear}-${String(startMonth).padStart(2, '0')}`;
  let endYM = `${endYear}-${String(endMonth).padStart(2, '0')}`;

  fetch('/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      country: combo.country,
      purpose: combo.purpose,
      start_ym: startYM,
      end_ym: endYM,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      detailDiv.style.display = 'block';
      if (data.error) {
        detailDiv.innerHTML = `<div class='alert alert-danger'>리포트 생성 오류: ${data.error}</div>`;
        return;
      }
      // 한글 리포트 및 비교표 표시
      let html = `<div class="card p-3 mb-2"><pre style="white-space:pre-line;">${data.report}</pre></div>`;
      if (data.comparison && data.comparison.length > 0) {
        html += `<table class="table table-sm table-bordered"><thead><tr><th>연월</th><th>실제값</th><th>예측값</th></tr></thead><tbody>`;
        data.comparison.forEach((row) => {
          html += `<tr><td>${
            row.ym
          }</td><td>${row.actual.toLocaleString()}</td><td>${row.predicted.toLocaleString()}</td></tr>`;
        });
        html += `</tbody></table>`;
      }
      detailDiv.innerHTML = html;
    })
    .catch((err) => {
      detailDiv.style.display = 'block';
      detailDiv.innerHTML = `<div class='alert alert-danger'>리포트 API 호출 실패: ${err}</div>`;
    });
}
