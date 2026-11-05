---
layout: home
title: Home
---

<section id="about" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">About</h2>

    <p class="section-lead">
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

    <ul class="tags">
      <li>Multimodal Large Language Models</li>
      <li>Model Architecture Design</li>
      <li>Efficient AI</li>
    </ul>
  </div>
</section>

<section id="research" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">Research</h2>

    <ul class="research-grid">
      <li>
        <a class="research-card" href="{{ '/#publications' | relative_url }}">
          <svg class="research-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="4" width="17" height="16" rx="1.5"/><path d="M7.75 4v16M12 4v16M16.25 4v16"/></svg>
          <svg class="research-go" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg>
          <h3>Efficient Architectures</h3>
          <p>1-bit attention and linear attention.</p>
        </a>
      </li>
      <li>
        <a class="research-card" href="{{ '/projects/omnimodal-token-pruning/' | relative_url }}">
          <svg class="research-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M13 3 5 13.5h6L10 21l8-10.5h-6z"/></svg>
          <svg class="research-go" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg>
          <h3>Efficient Algorithms</h3>
          <p>Token pruning for omni-modal models, PEFT, and continual learning.</p>
        </a>
      </li>
      <li>
        <a class="research-card" href="{{ '/projects/missing-modality-avqa/' | relative_url }}">
          <svg class="research-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v4h-4"/></svg>
          <svg class="research-go" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg>
          <h3>Adaptive Algorithms</h3>
          <p>Robust reasoning even when a modality goes missing.</p>
        </a>
      </li>
    </ul>
  </div>
</section>

<section id="publications" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">Publications</h2>

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
              <span class="pub-venue">{{ pub.venue }}</span>
              <h4 class="pub-title">{{ pub.title }}</h4>

              {% if pub.authors %}
              <p class="pub-authors">{{ pub.authors | replace: 'J. Lim', '<span class="me">J. Lim</span>' }}</p>
              {% endif %}

              {% if pub.summary %}
              <p class="pub-summary">{{ pub.summary }}</p>
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
    <h2 class="section-title">Projects</h2>

    <ul class="card-list">
      {% for p in site.data.projects limit: 3 %}
        {% include project-card.html p=p compact=true %}
      {% endfor %}
    </ul>

    {% if site.data.projects.size > 3 %}
    <p class="section-more">
      <a href="{{ '/projects/' | relative_url }}">View all {{ site.data.projects.size }} projects &rarr;</a>
    </p>
    {% else %}
    <p class="section-more">
      <a href="{{ '/projects/' | relative_url }}">All projects &rarr;</a>
    </p>
    {% endif %}
  </div>
</section>

<section id="education" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">Education</h2>

    <ul class="card-list">
      {% for edu in site.data.education %}
      <li class="card">
        <h3 class="entry-title">{{ edu.degree }}</h3>
        <p class="entry-meta">{{ edu.school }}{% if edu.period %} &middot; {{ edu.period }}{% endif %}</p>
      </li>
      {% endfor %}
    </ul>
  </div>
</section>

<section id="skills" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">Skills</h2>

    <div class="card rows">
      <div>
        <div class="row-label">Programming Languages</div>
        <div class="row-value">Python</div>
      </div>
      <div>
        <div class="row-label">AI / ML Frameworks and Tools</div>
        <div class="row-value">PyTorch, TensorFlow, ONNX, Transformers, Accelerate, DeepSpeed</div>
      </div>
      <div>
        <div class="row-label">Infrastructure and Development Tools</div>
        <div class="row-value">Docker, Linux, Git</div>
      </div>
      <div>
        <div class="row-label">Languages</div>
        <div class="row-value">Korean, English</div>
      </div>
    </div>
  </div>
</section>
