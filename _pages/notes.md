---
layout: page
title: notes
permalink: /notes/
description: A growing collection of your cool notes.
nav: true
nav_order: 3
horizontal: false
---

<!-- pages/notes.md -->
<div class="notes">
{% assign notes_by_category = site.notes | group_by_exp: "item", "item.path | split: '/' | slice: 1, 1 | join: ''" %}
{% for category_group in notes_by_category %}
  {% if category_group.name == "" or category_group.name == nil %}{% continue %}{% endif %}
  <div class="mt-5 pt-5">
    <a id="{{ category_group.name | slugify }}" href=".#{{ category_group.name | slugify }}">
      <h2 class="category font-weight-bold text-capitalize" style="font-size: 1.875rem; padding-bottom: 0.25rem; border-bottom: 1px solid #E5E7EB; margin-bottom: 1rem;">{{ category_group.name }}</h2>
    </a>
  </div>

  {% assign notes_with_topics = "" | split: "" %}
  {% for note in category_group.items %}
    {% assign path_parts_size = note.path | split: '/' | size %}
    {% if path_parts_size > 3 %}
      {% assign notes_with_topics = notes_with_topics | push: note %}
    {% endif %}
  {% endfor %}

  {% if notes_with_topics.size > 0 %}
    {% assign topics = notes_with_topics | group_by_exp: "item", "item.path | split: '/' | slice: 2, 1 | join: ''" %}
    <div class="row row-cols-1 row-cols-sm-2 row-cols-lg-3 g-4">
      {% for topic in topics %}
        {% include topic_card.liquid topic=topic %}
      {% endfor %}
    </div>
  {% endif %}
{% endfor %}
</div>