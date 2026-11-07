/* Theme toggle, scroll reveal, and nav state. No dependencies. */
(function () {
  'use strict'

  /* --- Theme ------------------------------------------------------------ */
  // The initial class is set inline in <head> to avoid a flash of the wrong
  // theme; here we only handle the toggle and keep the choice.
  var root = document.documentElement
  var toggle = document.querySelector('.theme-toggle')

  var label = function () {
    if (!toggle) return
    toggle.setAttribute(
      'aria-label',
      root.classList.contains('dark') ? 'Switch to light theme' : 'Switch to dark theme'
    )
  }

  if (toggle) {
    // The markup ships one label; correct it for whichever theme actually loaded.
    label()

    toggle.addEventListener('click', function () {
      var dark = root.classList.toggle('dark')
      label()
      try {
        localStorage.setItem('theme', dark ? 'dark' : 'light')
      } catch (e) {
        /* private mode or blocked storage — the toggle still works this visit */
      }
    })
  }

  /* --- Language --------------------------------------------------------- */
  // The language is chosen in <head> before paint; here we sync the switch
  // and keep the reader's choice.
  var langBtns = document.querySelectorAll('[data-set-lang]')
  var syncLang = function () {
    var cur = root.getAttribute('data-lang')
    Array.prototype.forEach.call(langBtns, function (b) {
      b.setAttribute('aria-pressed', b.getAttribute('data-set-lang') === cur ? 'true' : 'false')
    })
  }
  syncLang()
  Array.prototype.forEach.call(langBtns, function (b) {
    b.addEventListener('click', function () {
      var lang = b.getAttribute('data-set-lang')
      root.setAttribute('data-lang', lang)
      root.lang = lang
      syncLang()
      try { localStorage.setItem('lang', lang) } catch (e) {}
    })
  })

  /* --- Project filter ---------------------------------------------------- */
  // The chips ship hidden so the grid reads complete without this script.
  Array.prototype.forEach.call(document.querySelectorAll('[data-pgrid]'), function (grid) {
    var bar = grid.querySelector('.pfilter')
    if (!bar) return
    var chips = bar.querySelectorAll('[data-filter]')
    var cards = grid.querySelectorAll('.pcard')
    bar.hidden = false
    Array.prototype.forEach.call(chips, function (chip) {
      chip.addEventListener('click', function () {
        var f = chip.getAttribute('data-filter')
        Array.prototype.forEach.call(chips, function (c) { c.setAttribute('aria-pressed', c === chip ? 'true' : 'false') })
        Array.prototype.forEach.call(cards, function (card) {
          card.hidden = !(f === 'all' || card.getAttribute('data-cat') === f)
        })
      })
    })
  })

  /* --- Reveal on scroll -------------------------------------------------- */
  var revealables = document.querySelectorAll('.reveal')

  if (!('IntersectionObserver' in window)) {
    // Old browser: just show everything.
    Array.prototype.forEach.call(revealables, function (el) { el.classList.add('in') })
  } else {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return
        entry.target.classList.add('in')
        observer.unobserve(entry.target)
      })
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 })

    Array.prototype.forEach.call(revealables, function (el) { observer.observe(el) })
  }

  /* --- Navbar hairline on scroll ----------------------------------------- */
  var nav = document.querySelector('.nav')
  if (nav) {
    var onScroll = function () {
      nav.classList.toggle('scrolled', window.scrollY > 8)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
  }

  /* --- Highlight the section currently in view ---------------------------- */
  var navLinks = document.querySelectorAll('.nav-links a[href*="#"]')
  var sections = []

  Array.prototype.forEach.call(navLinks, function (link) {
    var id = link.getAttribute('href').split('#')[1]
    var section = id && document.getElementById(id)
    if (section) sections.push({ link: link, section: section })
  })

  if (sections.length && 'IntersectionObserver' in window) {
    var visible = new Set()

    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) visible.add(entry.target)
        else visible.delete(entry.target)
      })

      // Topmost visible section wins, so the label matches what you read.
      var current = sections.filter(function (s) { return visible.has(s.section) })[0]
      sections.forEach(function (s) {
        s.link.classList.toggle('active', !!current && s === current)
      })
    }, { rootMargin: '-45% 0px -50% 0px' })

    sections.forEach(function (s) { spy.observe(s.section) })
  }
})()
