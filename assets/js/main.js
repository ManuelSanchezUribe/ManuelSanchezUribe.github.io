/* ==========================================================================
   ACADEMIC WEBSITE INTERACTIVE ENGINE - PROF. MANUEL A. SÁNCHEZ
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initHeroCanvas();
  initPublicationsFilter();
  initBibtexCopy();
  initSmoothScroll();
});

/* --------------------------------------------------------------------------
   1. THEME SWITCHER (DARK / LIGHT MODE)
   -------------------------------------------------------------------------- */
function initThemeToggle() {
  const toggleBtn = document.getElementById('theme-toggle');
  if (!toggleBtn) return;

  const currentTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', currentTheme);
  updateThemeIcon(currentTheme);

  toggleBtn.addEventListener('click', () => {
    const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
  });
}

function updateThemeIcon(theme) {
  const icon = document.querySelector('#theme-toggle i');
  if (!icon) return;
  if (theme === 'dark') {
    icon.className = 'fas fa-sun';
  } else {
    icon.className = 'fas fa-moon';
  }
}

/* --------------------------------------------------------------------------
   2. INTERACTIVE HERO CANVAS (LORENZ ATTRACTOR / WAVE MESH)
   -------------------------------------------------------------------------- */
function initHeroCanvas() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width, height;

  function resize() {
    width = canvas.width = canvas.parentElement.offsetWidth;
    height = canvas.height = canvas.parentElement.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Lorenz Attractor Parameters
  let x = 0.1, y = 0, z = 0;
  const sigma = 10, rho = 28, beta = 8 / 3;
  const dt = 0.008;

  const points = [];
  const maxPoints = 850;

  let mouseX = 0, mouseY = 0;
  window.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 0.5;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 0.5;
  });

  function draw() {
    ctx.clearRect(0, 0, width, height);

    // Compute Lorenz step
    const dx = sigma * (y - x) * dt;
    const dy = (x * (rho - z) - y) * dt;
    const dz = (x * y - beta * z) * dt;

    x += dx;
    y += dy;
    z += dz;

    // Scale & center for canvas
    const scale = Math.min(width, height) / 65;
    const cx = width / 2 + mouseX * 80;
    const cy = height / 2 + 30 + mouseY * 80;

    // Rotate points slightly
    const px = x * Math.cos(mouseX) - z * Math.sin(mouseX);
    const py = y;
    const pz = x * Math.sin(mouseX) + z * Math.cos(mouseX);

    points.push({
      x: cx + px * scale,
      y: cy + (py - 15) * scale,
      z: pz
    });

    if (points.length > maxPoints) {
      points.shift();
    }

    // Draw Attractor Curve with Gradient
    if (points.length > 1) {
      ctx.lineWidth = 1.8;
      for (let i = 1; i < points.length; i++) {
        const p1 = points[i - 1];
        const p2 = points[i];

        const alpha = i / points.length;
        const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
        const color = isDark 
          ? `rgba(56, 189, 248, ${alpha * 0.7})` 
          : `rgba(2, 132, 199, ${alpha * 0.6})`;

        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    requestAnimationFrame(draw);
  }

  draw();
}

/* --------------------------------------------------------------------------
   3. PUBLICATIONS LIVE SEARCH & CATEGORY FILTERING
   -------------------------------------------------------------------------- */
function initPublicationsFilter() {
  const searchInput = document.getElementById('pub-search');
  const filterPills = document.querySelectorAll('.pill-btn');
  const pubCards = document.querySelectorAll('.pub-card');

  if (!pubCards.length) return;

  let activeCategory = 'all';
  let searchQuery = '';

  function filterItems() {
    pubCards.forEach(card => {
      const text = card.textContent.toLowerCase();
      const category = card.dataset.category || 'all';

      const matchesSearch = text.includes(searchQuery.toLowerCase());
      const matchesCategory = activeCategory === 'all' || category.includes(activeCategory);

      if (matchesSearch && matchesCategory) {
        card.style.display = 'grid';
      } else {
        card.style.display = 'none';
      }
    });
  }

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      searchQuery = e.target.value;
      filterItems();
    });
  }

  filterPills.forEach(pill => {
    pill.addEventListener('click', () => {
      filterPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      activeCategory = pill.dataset.filter;
      filterItems();
    });
  });
}

/* --------------------------------------------------------------------------
   4. BIBTEX GENERATOR & TOAST NOTIFICATION
   -------------------------------------------------------------------------- */
function initBibtexCopy() {
  document.addEventListener('click', (e) => {
    if (e.target.closest('.btn-bibtex')) {
      const btn = e.target.closest('.btn-bibtex');
      const bibtexData = btn.getAttribute('data-bibtex');

      if (bibtexData) {
        navigator.clipboard.writeText(bibtexData).then(() => {
          showToast('BibTeX copiado al portapapeles!');
        }).catch(err => {
          showToast('Error al copiar BibTeX');
        });
      }
    }
  });
}

function showToast(message) {
  let toast = document.getElementById('toast-notification');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast-notification';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }

  toast.innerHTML = `<i class="fas fa-check-circle" style="color: var(--accent-green)"></i> ${message}`;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 2500);
}

/* --------------------------------------------------------------------------
   5. SMOOTH SCROLLING & ACTIVE SECTION HIGHLIGHT
   -------------------------------------------------------------------------- */
function initSmoothScroll() {
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('section[id]');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 100;
      if (window.scrollY >= sectionTop) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}
