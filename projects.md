---
layout: page
title: Projects
permalink: /projects/
description: Selected research and engineering projects.
---

<section class="section reveal">
  <div class="wrap">
    <ul class="card-list">
      {% for p in site.data.projects %}
      <li class="card">
        <h2 class="entry-title">{{ p.title }}</h2>

        <p class="entry-meta">
          {%- if p.period %}{{ p.period }}{% endif -%}
          {%- if p.period and p.role %} &middot; {% endif -%}
          {%- if p.role %}{{ p.role }}{% endif -%}
        </p>

        {% if p.highlights %}
        <div class="entry-body">
          <ul>
            {% for h in p.highlights %}<li>{{ h }}</li>{% endfor %}
          </ul>
        </div>
        {% endif %}

        {% if p.stack %}
        <ul class="tags">
          {% for s in p.stack %}<li>{{ s }}</li>{% endfor %}
        </ul>
        {% endif %}

        {% if p.code or p.demo or p.paper %}
        <div class="pub-actions">
          {% if p.code %}<a class="btn btn-ghost" href="{{ p.code }}" rel="noopener">Code</a>{% endif %}
          {% if p.demo %}<a class="btn btn-ghost" href="{{ p.demo }}" rel="noopener">Demo</a>{% endif %}
          {% if p.paper %}<a class="btn btn-ghost" href="{{ p.paper }}" rel="noopener">Paper</a>{% endif %}
        </div>
        {% endif %}
      </li>
      {% endfor %}
    </ul>
  </div>
</section>
