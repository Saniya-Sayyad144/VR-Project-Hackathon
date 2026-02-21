// auth-check.js
window.addEventListener('DOMContentLoaded', function() {
    fetch('/check-auth', {
        method: 'GET',
        credentials: 'include'
    })
    .then(response => {
        if (response.status === 200) {
            // User is authenticated, allow page to load
            return;
        } else {
            // User is not authenticated, redirect to login
            window.location.href = '/login-page';
        }
    })
    .catch(() => {
        // Error checking auth, redirect to login
        window.location.href = '/login-page';
    });
});

