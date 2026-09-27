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
        {% include project-card.html p=p %}
      {% endfor %}
    </ul>
  </div>
</section>
