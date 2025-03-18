# Usa una imagen base de nginx
FROM nginx:alpine

# Copia los archivos del frontend al directorio de nginx
COPY index.html /usr/share/nginx/html/
COPY script.js /usr/share/nginx/html/

# Expone el puerto 80
EXPOSE 80