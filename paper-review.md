---
layout: archive
title: "Paper Review"
permalink: /paper-review/
author_profile: true
toc: true
---

{% assign entries_layout = page.entries_layout | default: 'list' %}

<style>
  /* 제목 스타일 좀 더 예쁘게 (선택사항) */
  summary {
    cursor: pointer;
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 10px;
    outline: none;
    padding: 10px 0;
    border-bottom: 1px solid #eaeaea;
  }
  summary:hover {
    color: #2c3e50; /* 마우스 올렸을 때 색상 */
  }
  /* 화살표 숨기기 (원하면 제거 가능) */
  details > summary {
    list-style: none;
  }
  details > summary::-webkit-details-marker {
    display: none;
  }
</style>

<details>
  <summary>📂 Generative Model </summary>
  <div markdown="1">
  {% for post in site.categories['Generative Model'] %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
  </div>
</details>

<br>

<details>
  <summary>📂 Multi-Modal Learning </summary>
  <div markdown="1">
  {% for post in site.categories['Multi-Modal Learning'] %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
  </div>
</details>

<br>

<details>
  <summary>📂 Vision Transformer </summary>
  <div markdown="1">
  {% for post in site.categories['Vision Transformer'] %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
  </div>
</details>

<br>

<details>
  <summary>📂 Object detection </summary>
  <div markdown="1">
  {% for post in site.categories['Object detection'] %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
  </div>
</details>