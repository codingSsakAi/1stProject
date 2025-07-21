let countryList = [], purposeList = [];
let latestCovidVisual = 'show'; // 코로나 구간 시각화 on/off (초기: 표시)
let allPurposes = []; // 목적 전체 목록 저장 (체크박스 생성용)
let selectedPurposeTraces = {};

// 목적에서 '기타' 제거 함수
function filterPurposes(list) {
    return list.filter(v => v !== "기타");
}

// 상단 메뉴 토글
document.getElementById('menu-predict').onclick = function(e) {
    e.preventDefault();
    document.getElementById('predict-section').style.display = 'block';
    document.getElementById('news-section').style.display = 'none';
};
document.getElementById('menu-news').onclick = function(e) {
    e.preventDefault();
    document.getElementById('predict-section').style.display = 'none';
    document.getElementById('news-section').style.display = 'block';
    loadNewsList(1);
};

// 코로나 구간 표시/미표시 드롭다운 이벤트
document.getElementById('covid-visual').onchange = function() {
    latestCovidVisual = this.value;
    document.getElementById('compare-btn').click(); // 그래프 다시 그림
};

function setSelectOptions(id, options, defaultValue, withNone) {
    let sel = document.getElementById(id);
    sel.innerHTML = '';
    if (withNone) {
        let opt = document.createElement('option');
        opt.value = "미선택";
        opt.text = "미선택";
        sel.appendChild(opt);
    }
    options.forEach(v=>{
        let opt = document.createElement('option');
        opt.value = v;
        opt.text = v;
        sel.appendChild(opt);
    });
    sel.value = defaultValue || options[0];
}

// 국가/목적 불러오기
fetch('/api/countries').then(res=>res.json()).then(list=>{
    countryList = list;
    setSelectOptions('country1', countryList, countryList[0], false);
});
fetch('/api/purposes').then(res=>res.json()).then(list=>{
    // '기타' 제거, '전체'는 유지
    purposeList = filterPurposes(list);
    setSelectOptions('purpose1', purposeList, purposeList[0], false);
    // 목적 전체 목록 저장(그래프에 사용)
    allPurposes = purposeList.filter(p => p !== "전체");
});

// 연/월 셋팅
const minYear = 2005, maxYear = 2025, minMonth = 1, maxMonth = 5;
setYearMonthSelect('start-year', 'start-month', 'end-year', 'end-month', minYear, minMonth, maxYear, maxMonth);

document.getElementById('compare-btn').onclick = function(){
    let combos = [
        {country: document.getElementById('country1').value, purpose: document.getElementById('purpose1').value}
    ];
    let startYear = parseInt(document.getElementById('start-year').value);
    let startMonth = parseInt(document.getElementById('start-month').value);
    let endYear = parseInt(document.getElementById('end-year').value);
    let endMonth = parseInt(document.getElementById('end-month').value);
    let startYM = `${startYear}-${String(startMonth).padStart(2,'0')}`;
    let endYM = `${endYear}-${String(endMonth).padStart(2,'0')}`;
    let covidOption = document.getElementById('covid-option').value;

    // 전체 선택 시: 목적별+전체 여러 개 combo를 만듦
    let purpose = document.getElementById('purpose1').value;
    let fetchCombos = [];
    if(purpose === "전체") {
        fetchCombos.push({country: combos[0].country, purpose: "전체"});
        allPurposes.forEach(p=>{
            fetchCombos.push({country: combos[0].country, purpose: p});
        });
    } else {
        fetchCombos = combos;
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            combos: fetchCombos,
            start_ym: startYM,
            end_ym: endYM,
            covid_option: covidOption
        })
    }).then(res=>res.json()).then(res=>{
        drawGraphWithCheckbox(res.results, 'compare-graph', 'compare-title', fetchCombos);
    });
};

function toKoMonth(x){
    let [yy, mm] = x.split('-');
    return `${yy}년 ${parseInt(mm)}월`;
}

// 목적별 체크박스/흐리게 기능 포함 그래프 그리기
function drawGraphWithCheckbox(results, divId, titleId, combos) {
    let traces = [];
    let colors = [
        '#1b77c2',  // 전체 (파랑)
        '#43aa8b',  // 공용 (초록)
        '#2196f3',  // 관광 (스카이블루)
        '#bc5090',  // 상용 (보라핑크)
        '#ffb703',  // 유학연수 (노랑)
    ];
    let allY = [];
    let purposeNames = [];
    let countryName = results[0]?.country || "";
    let mainPurpose = combos.length > 1 ? "전체" : results[0]?.purpose || "";
    let subPurposeStr = "";

    // 목적 리스트 만들기
    results.forEach((r, idx)=>{
        if (!r || r.error) return;
        let yArr = (r.values||[]).map(v=>isNaN(v)||v==null?0:v);
        allY = allY.concat(yArr);
        if(r.purpose !== "전체") purposeNames.push(r.purpose);

        // *** 목적명 customdata 및 hovertemplate, hoverlabel (테두리색 적용) ***
        traces.push({
            x: r.yms,
            y: yArr,
            name: r.purpose,
            mode: 'lines+markers',
            line: {color: colors[idx % colors.length], width: 3, dash: 'solid'},
            marker: {color: colors[idx % colors.length], size: 9},
            opacity: (selectedPurposeTraces[r.purpose]===false)?0.2:1,
            customdata: Array(r.yms.length).fill(r.purpose), // 목적명 배열
            hovertemplate: 
                '<b>목적: %{customdata}</b><br>' + // 목적명 강조
                '%{x|%Y년 %m월}<br>' +              // x축(연월) 한글 표시
                '입국자: %{y:,}<extra></extra>',    // 입국자수 콤마
            hoverlabel: {
                font: {size: 19, family: 'Noto Sans KR, Malgun Gothic, sans-serif'},
                bgcolor: "#fff",
                bordercolor: colors[idx % colors.length] // 목적별 선 색상과 동일!
            }
        });
    });

    let yMax = Math.max(...allY, 0);
    let yaxis_range = yMax === 0 ? [0, 10000] : [0, Math.ceil(yMax * 1.1)];

    // 타이틀 구성
    let globeIcon = '<img src="/static/images/globe.png" alt="지구본" style="width:22px;height:22px;vertical-align:middle;margin-bottom:3px;margin-right:4px;">';
    if(mainPurpose === "전체" && purposeNames.length > 0) {
        subPurposeStr = `<span style="font-size:0.95em;color:#888;">(${purposeNames.join(', ')})</span>`;
    }
    let titleHtml = `${globeIcon}: ${countryName} | 목적: ${mainPurpose} ${subPurposeStr}`;
    document.getElementById(titleId).innerHTML = titleHtml;

    // 목적 체크박스(전체 제외)
    if(combos.length > 1){
        let html = "";
        // 1. 전체 포함! 목적 순서: 전체, 공용, 관광, 상용, 유학연수 ...
        let allNames = ['전체', ...purposeNames];
        allNames.forEach(name=>{
            const checked = (selectedPurposeTraces[name]!==false)?'checked':'';
            html += `<label style="margin-right:22px;font-weight:500;cursor:pointer;">
                        <input type="checkbox" value="${name}" ${checked}> ${name}
                    </label>`;
        });
        document.getElementById("purpose-checkboxes").innerHTML = html;
        document.querySelectorAll("#purpose-checkboxes input[type=checkbox]").forEach(chk=>{
            chk.onchange = function(){
                selectedPurposeTraces[this.value] = this.checked;
                drawGraphWithCheckbox(results, divId, titleId, combos);
            }
        });
    } else {
        document.getElementById("purpose-checkboxes").innerHTML = "";
        selectedPurposeTraces = {};
    }

    // 그래프 데이터 없으면 안내 표시
    if(traces.every(t=>!t.y.some(v=>v>0))) {
        document.getElementById(divId).innerHTML =
            "<div style='text-align:center;padding:100px 0;color:#aaa;font-size:1.5em;'>데이터가 없습니다.</div>";
        return;
    }

    // 코로나 구간
    let shapes = [];
    if (latestCovidVisual === 'show' && traces[0] && traces[0].x) {
        shapes = [{
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: "2020-03",
            x1: "2022-10",
            y0: 0, y1: 1,
            fillcolor: 'rgba(255,0,0,0.13)',
            line: {width:0}
        }];
    }

    Plotly.newPlot(divId, traces, {
        xaxis: {
            title: '월',
            tickangle: -45,
            tickvals: traces[0]?.x,
            ticktext: traces[0]?.x.map(toKoMonth)
        },
        yaxis: {
            title: '입국자 수',
            rangemode: 'tozero',
            tickformat: ',d',
            range: yaxis_range,
            automargin:true
        },
        margin: { t:30, b:130, l:60, r:10 },
        height: 500,
        legend: {orientation: "h", yanchor: "bottom", y: -0.45, xanchor: "center", x: 0.5, font: { size: 16 }},
        font: {family: 'Noto Sans KR, Malgun Gothic, sans-serif'},
        shapes: shapes
    }, {responsive: true});
}

// --------- 이하 기존 뉴스 스크랩 코드 동일 ----------
function setYearMonthSelect(sy, sm, ey, em, minY, minM, maxY, maxM) {
    let years = [];
    for (let y = minY; y <= maxY; y++) years.push(y);
    let months = [];
    for (let m = 1; m <= 12; m++) months.push(m);
    [sy, ey].forEach(id=>{
        let sel = document.getElementById(id);
        sel.innerHTML = '';
        years.forEach(y=>{
            let opt = document.createElement('option');
            opt.value = y;
            opt.text = y + '년';
            sel.appendChild(opt);
        });
    });
    [sm, em].forEach(id=>{
        let sel = document.getElementById(id);
        sel.innerHTML = '';
        months.forEach(m=>{
            let opt = document.createElement('option');
            opt.value = m;
            opt.text = m + '월';
            sel.appendChild(opt);
        });
    });
}

function loadNewsList(page=1) {
    const ul = document.getElementById('news-list');
    ul.innerHTML = "<li class='list-group-item'>뉴스를 불러오는 중...</li>";
    fetch('/api/news?page=' + page)
        .then(res => res.json())
        .then(data => {
            ul.innerHTML = '';
            if(data.news.length === 0) {
                ul.innerHTML = "<li class='list-group-item'>뉴스가 없습니다.</li>";
                document.getElementById('news-pagination').innerHTML = '';
                return;
            }
            data.news.forEach(item => {
            ul.innerHTML += `<li class="list-group-item">
                <a href="${item.link}" target="_blank">${item.title.replace(/<b>|<\/b>/g, '')}</a>
                <span class="badge bg-secondary">${item.pubDate}</span>
            </li>`;
            });
            renderNewsPagination(data.total, data.page, data.page_size);
        })
        .catch(()=>{
            ul.innerHTML = "<li class='list-group-item'>뉴스를 불러오지 못했습니다.</li>";
            document.getElementById('news-pagination').innerHTML = '';
        });
}
function renderNewsPagination(total, page, pageSize) {
    const totalPages = Math.ceil(total / pageSize);
    let html = '';
    for(let i=1;i<=totalPages;i++) {
        html += `<li class="page-item${i===page?' active':''}"><a class="page-link" href="#" onclick="loadNewsList(${i});return false;">${i}</a></li>`;
    }
    document.getElementById('news-pagination').innerHTML = html;
}
document.addEventListener('DOMContentLoaded', function(){
    document.getElementById('compare-btn').click();
});
