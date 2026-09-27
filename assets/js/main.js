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
