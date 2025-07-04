# BioMedGPS Explainer Documentation

This directory contains the complete documentation for BioMedGPS Explainer, designed to be deployed as a GitHub Pages site.

## 📁 Directory Structure

```
docs/
├── index.html                 # Main landing page
├── user-guide.html            # Complete user guide
├── api.html                   # API documentation
├── examples.html              # Examples and demos
├── model-usage.html           # Model usage guide
├── examples/
│   └── asthma-analysis.html   # Sample analysis report
├── assets/
│   ├── css/
│   │   ├── style.css         # Main styles
│   │   ├── docs.css          # Documentation styles
│   │   └── report.css        # Report styles
│   └── js/
│       ├── script.js         # Main JavaScript
│       └── report.js         # Report functionality
├── _config.yml               # GitHub Pages configuration
└── README.md                 # This file
```

## 🚀 GitHub Pages Setup

### 1. Enable GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" section
3. Select "Deploy from a branch"
4. Choose "main" branch and "/docs" folder
5. Click "Save"

### 2. Configure Repository

Update the following files with your actual repository information:

- `_config.yml`: Update URLs and repository names
- All HTML files: Update GitHub links
- `index.html`: Update repository URL in navigation

### 3. Custom Domain (Optional)

If you have a custom domain:

1. Add your domain to the `_config.yml` file
2. Create a `CNAME` file in the docs directory with your domain
3. Configure DNS settings with your domain provider

## 🎨 Customization

### Colors and Branding

The site uses a modern color scheme that can be customized in `assets/css/style.css`:

```css
:root {
  --primary-color: #2563eb;
  --secondary-color: #667eea;
  --accent-color: #764ba2;
  --text-color: #1f2937;
  --background-color: #ffffff;
}
```

### Adding New Pages

1. Create a new HTML file in the docs directory
2. Include the standard navigation and footer
3. Add the page to the navigation menu in all HTML files
4. Update the table of contents in the sidebar

### Adding New Examples

1. Create a new HTML file in the `examples/` directory
2. Follow the structure of `asthma-analysis.html`
3. Add the example to `examples.html`
4. Update any relevant links

## 📊 Features

### Interactive Elements

- **Responsive Design**: Works on all device sizes
- **Smooth Navigation**: Tab-based navigation for reports
- **Interactive Charts**: Plotly.js integration for data visualization
- **Code Copy**: One-click code copying functionality
- **Search**: Full-text search across documentation
- **Dark Mode**: Automatic dark mode support

### Documentation Features

- **Table of Contents**: Auto-generated navigation
- **Syntax Highlighting**: Code blocks with syntax highlighting
- **Responsive Tables**: Mobile-friendly data tables
- **Print Styles**: Optimized for printing
- **SEO Optimized**: Meta tags and structured data

## 🔧 Development

### Local Development

1. Install Jekyll (if using Jekyll features):
   ```bash
   gem install jekyll bundler
   bundle install
   ```

2. Serve locally:
   ```bash
   jekyll serve --source docs
   ```

3. Or use a simple HTTP server:
   ```bash
   python -m http.server 8000
   # or
   npx serve docs
   ```

### File Organization

- **HTML Files**: Main documentation pages
- **CSS Files**: Styling and layout
- **JavaScript Files**: Interactivity and functionality
- **Examples**: Sample reports and demonstrations
- **Assets**: Images, icons, and other resources

## 📝 Content Guidelines

### Writing Style

- Use clear, concise language
- Include code examples where appropriate
- Provide step-by-step instructions
- Use consistent terminology

### Code Examples

- Include complete, runnable examples
- Add comments to explain complex code
- Use syntax highlighting
- Provide both Python and CLI examples

### Images and Media

- Optimize images for web (compress, resize)
- Use descriptive alt text
- Include captions for complex diagrams
- Store images in `assets/images/`

## 🚀 Deployment

### Automatic Deployment

GitHub Pages automatically deploys when you push to the main branch. The site will be available at:
`https://your-username.github.io/biomedgps-explainer`

### Manual Deployment

If you need to deploy manually:

1. Build the site (if using Jekyll):
   ```bash
   jekyll build --source docs
   ```

2. Upload the `_site` directory to your web server

## 🔍 SEO and Analytics

### Search Engine Optimization

- Meta tags are included in all pages
- Structured data for better search results
- Sitemap generation
- Robots.txt configuration

### Analytics

To add Google Analytics:

1. Get your tracking ID from Google Analytics
2. Update `_config.yml` with your ID
3. Create `assets/js/analytics.js` with tracking code

## 🐛 Troubleshooting

### Common Issues

1. **Links not working**: Check that all paths are relative to the docs directory
2. **Styles not loading**: Verify CSS file paths and check for syntax errors
3. **JavaScript errors**: Check browser console for errors
4. **Images not displaying**: Ensure image paths are correct

### Performance Optimization

- Minify CSS and JavaScript files
- Optimize images (compress, use WebP format)
- Enable gzip compression on your server
- Use CDN for external libraries

## 📚 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [CSS Grid Guide](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [JavaScript Best Practices](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)

## 🤝 Contributing

To contribute to the documentation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This documentation is part of the BioMedGPS Explainer project and follows the same license terms. 