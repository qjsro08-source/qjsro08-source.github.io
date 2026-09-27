---
layout: home
title: Home
---

<section id="about" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">About</h2>

    <p class="section-lead">
      I am an M.S. student in Applied AI at SeoulTech, focusing on multimodal
      large language models (MLLMs) and efficient AI, with particular interests
      in model architecture design and optimization.
    </p>

    <ul class="tags">
      <li>Multimodal Large Language Models</li>
      <li>Model Architecture Design</li>
      <li>Efficient AI</li>
    </ul>
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

<section id="publications" class="section reveal">
  <div class="wrap">
    <h2 class="section-title">Publications</h2>

    {% for group in site.data.publications %}
    <div class="pub-group">
      <h3 class="pub-group-title">{{ group.group }}</h3>

      <ul class="card-list">
        {% for pub in group.items %}
        <li class="card">
          <span class="pub-venue">{{ pub.venue }}</span>
          <h4 class="pub-title">{{ pub.title }}</h4>

          {% if pub.authors %}
          <p class="pub-authors">{{ pub.authors | replace: 'J. Lim', '<span class="me">J. Lim</span>' }}</p>
          {% endif %}

          {% if pub.paper or pub.code or pub.project %}
          <div class="pub-actions">
            {% if pub.paper %}<a class="btn btn-ghost" href="{{ pub.paper }}" rel="noopener">Paper</a>{% endif %}
            {% if pub.code %}<a class="btn btn-ghost" href="{{ pub.code }}" rel="noopener">Code</a>{% endif %}
            {% if pub.project %}<a class="btn btn-ghost" href="{{ pub.project }}" rel="noopener">Project</a>{% endif %}
          </div>
          {% endif %}
        </li>
        {% endfor %}
      </ul>
    </div>
    {% endfor %}
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
        <div class="row-value">PyTorch, TensorFlow, ONNX</div>
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
