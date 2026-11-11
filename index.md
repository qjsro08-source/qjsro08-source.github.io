---
layout: home
title: Home
---

<section id="about" class="section reveal">
  <div class="wrap">
    <h2 class="section-title"><span data-l="en">About</span><span data-l="ko">소개</span></h2>

    <p class="section-lead" data-l="en">
      I am an M.S. student in Applied AI at SeoulTech, primarily focusing on
      multimodal large language models (MLLMs). My research aims to develop
      efficient MLLMs that reduce computational costs while preserving strong
      performance, with particular interests in model architecture design and
      optimization. I have also conducted research on 3D reconstruction and
      time-series data, which has broadened my perspective on how AI can
      perceive and model different aspects of the real world. Through these
      experiences, my broader research goal is to develop AI systems that can
      understand and reason about the world through diverse modalities.
    </p>
    <p class="section-lead" data-l="ko">
      서울과학기술대학교 인공지능응용학과 석사과정에서 멀티모달 대형 언어 모델(MLLM)을
      중심으로 연구하고 있습니다. 강한 성능은 유지하면서 연산 비용을 줄이는 효율적인 MLLM을 만드는 것이
      목표이며, 특히 모델 구조 설계와 최적화에 관심이 많습니다. 3D 복원과 시계열 데이터 연구도 진행하며
      AI가 실제 세계의 여러 측면을 어떻게 인식하고 모델링하는지에 대한 시야를 넓혔습니다. 이러한 경험을
      바탕으로, 다양한 모달리티를 통해 세상을 이해하고 추론하는 AI 시스템을 만드는 것이 더 큰 연구 목표입니다.
    </p>

    <ul class="tags">
      <li>Multimodal Large Language Models</li>
      <li>Model Architecture Design</li>
      <li>Efficient AI</li>
    </ul>
  </div>
</section>

<section id="research" class="section reveal">
  <div class="wrap">
    <h2 class="section-title"><span data-l="en">Research</span><span data-l="ko">연구</span></h2>

    <ul class="research-grid">
      <li>
        <a class="research-card" href="{{ '/projects/spike-dice-attention/' | relative_url }}">
          <span class="research-badge" aria-hidden="true"><svg class="research-icon" viewBox="0 0 24 24"><path d="M12 3.5 20.5 8 12 12.5 3.5 8z" class="ri-fill"/><path d="M3.5 12 12 16.5 20.5 12"/><path d="M3.5 16 12 20.5 20.5 16"/></svg></span>
          <svg class="research-go" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg>
          <h3>Efficient Architectures</h3>
          <p><span data-l="en">1-bit attention and linear attention.</span><span data-l="ko">1-bit attention과 linear attention.</span></p>
        </a>
      </li>
      <li>
        <a class="research-card" href="{{ '/projects/omnimodal-token-pruning/' | relative_url }}">
          <span class="research-badge" aria-hidden="true"><svg class="research-icon" viewBox="0 0 24 24"><rect class="ri-on" x="3.50" y="3.50" width="4.5" height="4.5" rx="1.1"/><rect class="ri-off" x="9.75" y="3.50" width="4.5" height="4.5" rx="1.1"/><rect class="ri-on" x="16.00" y="3.50" width="4.5" height="4.5" rx="1.1"/><rect class="ri-off" x="3.50" y="9.75" width="4.5" height="4.5" rx="1.1"/><rect class="ri-on" x="9.75" y="9.75" width="4.5" height="4.5" rx="1.1"/><rect class="ri-off" x="16.00" y="9.75" width="4.5" height="4.5" rx="1.1"/><rect class="ri-on" x="3.50" y="16.00" width="4.5" height="4.5" rx="1.1"/><rect class="ri-off" x="9.75" y="16.00" width="4.5" height="4.5" rx="1.1"/><rect class="ri-on" x="16.00" y="16.00" width="4.5" height="4.5" rx="1.1"/></svg></span>
          <svg class="research-go" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg>
          <h3>Efficient Algorithms</h3>
          <p><span data-l="en">Token pruning for omni-modal models, PEFT, and continual learning.</span><span data-l="ko">옴니모달 모델의 token pruning, PEFT, continual learning.</span></p>
        </a>
      </li>
      <li>
        <a class="research-card" href="{{ '/projects/missing-modality-avqa/' | relative_url }}">
          <span class="research-badge" aria-hidden="true"><svg class="research-icon" viewBox="0 0 24 24"><circle cx="9" cy="12" r="5.5" class="ri-fill"/><circle cx="15" cy="12" r="5.5" class="ri-off"/></svg></span>
          <svg class="research-go" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg>
          <h3>Adaptive Algorithms</h3>
          <p><span data-l="en">Robust reasoning even when a modality goes missing.</span><span data-l="ko">모달리티가 빠져도 강건한 추론.</span></p>
        </a>
      </li>
    </ul>
  </div>
</section>

<section id="news" class="section reveal">
  <div class="wrap">
    <h2 class="section-title"><span data-l="en">News</span><span data-l="ko">소식</span></h2>

    {%- comment -%}
      The three newest items show; the rest sit in a panel that slides open
      from the button under the list (open by default without JavaScript).
    {%- endcomment -%}
    {%- assign news_top = site.data.news | slice: 0, 3 -%}
    {%- assign news_rest = site.data.news | slice: 3, 999 %}
    <ul class="news-list">
      {%- for n in news_top %}{% include news-item.html n=n %}{% endfor %}
    </ul>

    {%- if news_rest.size > 0 %}
    <div class="news-more" id="news-more">
      <div class="news-more-inner">
        <ul class="news-list">
          {%- for n in news_rest %}{% include news-item.html n=n %}{% endfor %}
        </ul>
      </div>
    </div>
    <button class="news-toggle" type="button" aria-expanded="false" aria-controls="news-more">
      <span class="news-toggle-more"><span data-l="en">Show {{ news_rest.size }} more</span><span data-l="ko">{{ news_rest.size }}개 더 보기</span></span>
      <span class="news-toggle-less"><span data-l="en">Show less</span><span data-l="ko">접기</span></span>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
    </button>
    {%- endif %}
  </div>
</section>

<section id="publications" class="section reveal">
  <div class="wrap">
    <h2 class="section-title"><span data-l="en">Publications</span><span data-l="ko">논문</span></h2>

    {%- comment -%}
      Every visible paper, regrouped by year, newest year first. Hidden
      groups (e.g. under review) stay in the data file but are skipped.
    {%- endcomment -%}
    {%- assign pubs = "" | split: "" -%}
    {%- for group in site.data.publications -%}
      {%- unless group.hidden -%}{%- assign pubs = pubs | concat: group.items -%}{%- endunless -%}
    {%- endfor -%}
    {%- assign by_year = pubs | group_by: "year" | sort: "name" | reverse %}

    <div class="pub-years">
      {% for yg in by_year %}
      <div class="pub-year">
        <h3 class="pub-year-label">{{ yg.name }}</h3>

        <ul class="pub-list">
          {% for pub in yg.items %}
          <li class="pub-item">
            <div class="pub-main">
              <div class="pub-tags">
                <span class="pub-venue">{{ pub.venue }}</span>
                {%- if pub.role == "first" %}<span class="pub-role is-first">First Author</span>
                {%- elsif pub.role == "co" %}<span class="pub-role">Co-Author</span>{% endif %}
              </div>
              <h4 class="pub-title">{{ pub.title }}</h4>

              {% if pub.authors %}
              <p class="pub-authors">{{ pub.authors | replace: 'J. Lim', '<span class="me">J. Lim</span>' }}</p>
              {% endif %}

              {% if pub.summary %}
              {% include t.html tag="p" class="pub-summary" ko=pub.summary_ko en=pub.summary %}
              {% endif %}

              {% if pub.paper or pub.code or pub.project or pub.poster or pub.video %}
              <p class="pub-links">
                {% if pub.paper %}<a href="{{ pub.paper }}" target="_blank" rel="noopener">Paper<span aria-hidden="true">↗</span></a>{% endif %}
                {% if pub.code %}<a href="{{ pub.code }}" target="_blank" rel="noopener">Code<span aria-hidden="true">↗</span></a>{% endif %}
                {% if pub.project %}<a href="{{ pub.project }}" target="_blank" rel="noopener">Project<span aria-hidden="true">↗</span></a>{% endif %}
                {% if pub.poster %}<a href="{% if pub.poster contains '://' %}{{ pub.poster }}{% else %}{{ pub.poster | relative_url }}{% endif %}" target="_blank" rel="noopener">Poster<span aria-hidden="true">↗</span></a>{% endif %}
                {% if pub.video %}<a href="{{ pub.video }}" target="_blank" rel="noopener">Video<span aria-hidden="true">↗</span></a>{% endif %}
              </p>
              {% endif %}
            </div>

            {% if pub.image %}
            {% assign fig = pub.image %}
            {% unless fig contains '://' %}{% assign fig = pub.image | relative_url %}{% endunless %}
            {% comment %}
              The thumbnail is unreadable at this size, so it opens something
              bigger: image_href when the full version is a different file (a
              poster PDF, say), otherwise the image itself.
            {% endcomment %}
            {% assign figlink = fig %}
            {% if pub.image_href %}
              {% assign figlink = pub.image_href %}
              {% unless figlink contains '://' %}{% assign figlink = pub.image_href | relative_url %}{% endunless %}
            {% endif %}
            <figure class="pub-figure">
              <a href="{{ figlink }}" target="_blank" rel="noopener"
                 aria-label="Open the full version for &ldquo;{{ pub.title | escape }}&rdquo;">
                <img src="{{ fig }}" alt="Figure from &ldquo;{{ pub.title | escape }}&rdquo;" loading="lazy">
              </a>
            </figure>
            {% endif %}
          </li>
          {% endfor %}
        </ul>
      </div>
      {% endfor %}
    </div>
  </div>
</section>

<section id="projects" class="section reveal">
  <div class="wrap">
    <div class="section-head">
      <h2 class="section-title"><span data-l="en">Projects</span><span data-l="ko">프로젝트</span></h2>
      <a class="section-head-link" href="{{ '/projects/' | relative_url }}"><span data-l="en">All projects</span><span data-l="ko">전체 프로젝트</span> <span aria-hidden="true">&rarr;</span></a>
    </div>
    <p class="section-sub"><span data-l="en">One page per project: the problem, the method and the results.</span><span data-l="ko">연구와 응용 프로젝트마다 한 페이지씩, 문제와 방법, 결과를 정리했습니다.</span></p>

    {% include project-grid.html limit=6 %}
  </div>
</section>

<section id="education" class="section reveal">
  <div class="wrap">
    <h2 class="section-title"><span data-l="en">Education</span><span data-l="ko">학력</span></h2>

    <ul class="card-list">
      {% for edu in site.data.education %}
      <li class="card">
        <h3 class="entry-title">{% include t.html ko=edu.degree_ko en=edu.degree %}</h3>
        <p class="entry-meta">{% include t.html ko=edu.school_ko en=edu.school %}{% if edu.period %} &middot; {{ edu.period }}{% endif %}</p>
      </li>
      {% endfor %}
    </ul>
  </div>
</section>

<section id="skills" class="section reveal">
  <div class="wrap">
    <h2 class="section-title"><span data-l="en">Skills</span><span data-l="ko">기술</span></h2>

    <div class="card rows">
      <div>
        <div class="row-label"><span data-l="en">Programming Languages</span><span data-l="ko">프로그래밍 언어</span></div>
        <div class="row-value">Python</div>
      </div>
      <div>
        <div class="row-label"><span data-l="en">AI / ML Frameworks and Tools</span><span data-l="ko">AI / ML 프레임워크 및 도구</span></div>
        <div class="row-value">PyTorch, TensorFlow, ONNX, Transformers, Accelerate, DeepSpeed</div>
      </div>
      <div>
        <div class="row-label"><span data-l="en">Infrastructure and Development Tools</span><span data-l="ko">인프라 및 개발 도구</span></div>
        <div class="row-value">Docker, Linux, Git</div>
      </div>
      <div>
        <div class="row-label"><span data-l="en">Languages</span><span data-l="ko">언어</span></div>
        <div class="row-value"><span data-l="en">Korean, English</span><span data-l="ko">한국어, 영어</span></div>
      </div>
    </div>
  </div>
</section>
