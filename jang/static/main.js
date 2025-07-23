document.addEventListener("DOMContentLoaded", function () {
    loadCountryAndPurposeOptions();
    setupMenu();

    document.getElementById("compare-btn").addEventListener("click", async function () {
        const country = document.getElementById("country1").value;
        const purpose = document.getElementById("purpose1").value;
        const startYear = document.getElementById("start-year").value;
        const startMonth = document.getElementById("start-month").value;
        const endYear = document.getElementById("end-year").value;
        const endMonth = document.getElementById("end-month").value;

        if (!startYear || !startMonth || !endYear || !endMonth) {
            alert("시작/종료 연도 및 월을 모두 입력하세요.");
            return;
        }

        const combos = [{ country, purpose }];
        const startYM = `${startYear}-${String(startMonth).padStart(2, '0')}`;
        const endYM = `${endYear}-${String(endMonth).padStart(2, '0')}`;

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ combos, start_ym: startYM, end_ym: endYM }),
            });
            const data = await res.json();
            drawCompareGraphs(data.results);
        } catch (err) {
            console.error("예측 요청 실패:", err);
            alert("예측 중 오류 발생");
        }
    });
});

function setupMenu() {
    const menuPredict = document.getElementById("menu-predict");
    const menuNews = document.getElementById("menu-news");
    const predictSection = document.getElementById("predict-section");
    const newsSection = document.getElementById("news-section");

    menuPredict.addEventListener("click", function () {
        predictSection.style.display = "block";
        newsSection.style.display = "none";
    });
    menuNews.addEventListener("click", function () {
        predictSection.style.display = "none";
        newsSection.style.display = "block";
        loadNews(1);
    });
}

async function loadCountryAndPurposeOptions() {
    const countrySel = document.getElementById("country1");
    const purposeSel = document.getElementById("purpose1");

    const countries = await (await fetch("/api/countries")).json();
    const purposes = await (await fetch("/api/purposes")).json();

    countries.forEach(c => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        countrySel.appendChild(opt);
    });

    purposes.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        purposeSel.appendChild(opt);
    });
}

function drawCompareGraphs(results) {
    const container = document.getElementById("compare-graph");
    container.innerHTML = "";

    results.forEach((res, idx) => {
        if (res.error) {
            const div = document.createElement("div");
            div.className = "alert alert-warning";
            div.textContent = `⚠️ ${res.country} / ${res.purpose}: ${res.error}`;
            container.appendChild(div);
            return;
        }

        const graphDiv = document.createElement("div");
        graphDiv.id = `graph-${idx}`;
        graphDiv.style.height = "400px";
        graphDiv.style.marginBottom = "30px";
        container.appendChild(graphDiv);

        const allYms = res.hist_yms.concat(res.yms);
        const allVals = res.hist_values.concat(Array(res.yms.length).fill(null));

        const trace_actual = {
            x: res.hist_yms,
            y: res.hist_values,
            mode: 'lines+markers',
            name: '실제값',
            line: { color: 'blue' }
        };

        const trace_pred = {
            x: res.yms,
            y: res.values,
            mode: 'lines+markers',
            name: '예측값',
            line: { color: 'red', dash: 'dot' }
        };

        const layout = {
            title: `${res.country} / ${res.purpose}`,
            xaxis: { title: '년월' },
            yaxis: { title: '입국자 수' },
            margin: { t: 50, b: 60 }
        };

        Plotly.newPlot(graphDiv.id, [trace_actual, trace_pred], layout);
    });
}

async function loadNews(page = 1) {
    const list = document.getElementById("news-list");
    const pagination = document.getElementById("news-pagination");
    list.innerHTML = "";
    pagination.innerHTML = "";

    const res = await fetch(`/api/news?page=${page}`);
    const data = await res.json();

    data.news.forEach(n => {
        const item = document.createElement("li");
        item.className = "list-group-item";
        item.innerHTML = `<strong>${n.pubDate}</strong> <a href="${n.link}" target="_blank">${n.title}</a><br><small>${n.description}</small>`;
        list.appendChild(item);
    });

    const totalPages = Math.ceil(data.total / data.page_size);
    for (let i = 1; i <= totalPages; i++) {
        const li = document.createElement("li");
        li.className = `page-item ${i === page ? 'active' : ''}`;
        li.innerHTML = `<a class="page-link" href="#">${i}</a>`;
        li.addEventListener("click", e => {
            e.preventDefault();
            loadNews(i);
        });
        pagination.appendChild(li);
    }
}
